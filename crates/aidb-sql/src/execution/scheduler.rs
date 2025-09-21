use crossbeam::deque::{Injector, Steal, Stealer, Worker};
use crossbeam::sync::WaitGroup;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

type Task = Box<dyn FnOnce() + Send + 'static>;

fn steal_work(
    worker_id: usize,
    worker: &Worker<Task>,
    injector: &Injector<Task>,
    stealers: &[Stealer<Task>],
) -> Option<Task> {
    if let Some(task) = worker.pop() {
        return Some(task);
    }
    loop {
        match injector.steal_batch_and_pop(worker) {
            Steal::Success(task) => return Some(task),
            Steal::Retry => continue,
            Steal::Empty => break,
        }
    }
    for (idx, stealer) in stealers.iter().enumerate() {
        if idx == worker_id {
            continue;
        }
        loop {
            match stealer.steal() {
                Steal::Success(task) => return Some(task),
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
    }
    None
}

fn worker_loop(
    worker_id: usize,
    worker: Worker<Task>,
    injector: Arc<Injector<Task>>,
    stealers: Arc<Vec<Stealer<Task>>>,
    notify: Arc<(Mutex<bool>, Condvar)>,
    shutdown: Arc<AtomicBool>,
) {
    while !shutdown.load(Ordering::Acquire) {
        if let Some(task) = steal_work(worker_id, &worker, &injector, &stealers) {
            task();
            continue;
        }

        if shutdown.load(Ordering::Acquire) {
            break;
        }

        let (lock, cvar) = &*notify;
        let mut signaled = lock.lock().unwrap();
        while !*signaled && !shutdown.load(Ordering::Acquire) {
            let wait_result = cvar
                .wait_timeout(signaled, Duration::from_millis(5))
                .unwrap();
            signaled = wait_result.0;
        }
        *signaled = false;
    }
}

pub(crate) struct TaskScheduler {
    injector: Arc<Injector<Task>>,
    notify: Arc<(Mutex<bool>, Condvar)>,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
    parallelism: usize,
}

impl TaskScheduler {
    pub(crate) fn new(parallelism: usize) -> Self {
        let parallelism = parallelism.max(1);
        let injector = Arc::new(Injector::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let notify = Arc::new((Mutex::new(false), Condvar::new()));

        let mut worker_deques = Vec::with_capacity(parallelism);
        let mut stealer_handles = Vec::with_capacity(parallelism);
        for _ in 0..parallelism {
            let worker = Worker::new_fifo();
            stealer_handles.push(worker.stealer());
            worker_deques.push(worker);
        }
        let stealers = Arc::new(stealer_handles);

        let mut workers = Vec::with_capacity(parallelism);
        for (idx, worker) in worker_deques.into_iter().enumerate() {
            let injector_clone = Arc::clone(&injector);
            let stealers_clone = Arc::clone(&stealers);
            let notify_clone = Arc::clone(&notify);
            let shutdown_clone = Arc::clone(&shutdown);
            let handle = thread::Builder::new()
                .name(format!("aidb-sql-worker-{idx}"))
                .spawn(move || {
                    worker_loop(
                        idx,
                        worker,
                        injector_clone,
                        stealers_clone,
                        notify_clone,
                        shutdown_clone,
                    );
                })
                .expect("failed to spawn scheduler worker");
            workers.push(handle);
        }

        Self {
            injector,
            notify,
            shutdown,
            workers,
            parallelism,
        }
    }

    pub(crate) fn execute<I, F>(&self, tasks: I)
    where
        I: IntoIterator<Item = F>,
        F: FnOnce() + Send + 'static,
    {
        let mut count = 0usize;
        let wait_group = WaitGroup::new();
        for task in tasks {
            count += 1;
            let guard = wait_group.clone();
            self.injector.push(Box::new(move || {
                task();
                drop(guard);
            }));
        }

        if count == 0 {
            return;
        }

        {
            let (lock, cvar) = &*self.notify;
            let mut flag = lock.lock().unwrap();
            *flag = true;
            cvar.notify_all();
        }

        wait_group.wait();
    }
}

impl Drop for TaskScheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        {
            let (lock, cvar) = &*self.notify;
            let mut flag = lock.lock().unwrap();
            *flag = true;
            cvar.notify_all();
        }
        // Push empty tasks to ensure workers exit steal loop promptly.
        for _ in 0..self.parallelism {
            self.injector.push(Box::new(|| {}));
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl fmt::Debug for TaskScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskScheduler")
            .field("parallelism", &self.parallelism)
            .finish()
    }
}
