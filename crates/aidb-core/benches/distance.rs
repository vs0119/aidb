use std::hint::black_box;
use std::time::Instant;

fn main() {
    let a: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..512).map(|i| (i * 2) as f32).collect();
    let iterations = 10_000;

    let start = Instant::now();
    let mut acc = 0.0;
    for _ in 0..iterations {
        acc += aidb_core::cosine_sim(black_box(&a), black_box(&b));
    }
    let elapsed = start.elapsed();
    println!(
        "cosine_sim 512 -> {:.2?} total ({:.2?} per iter)",
        elapsed,
        elapsed / iterations as u32
    );

    let start = Instant::now();
    let mut dist = 0.0;
    for _ in 0..iterations {
        dist += aidb_core::l2(black_box(&a), black_box(&b));
    }
    let elapsed = start.elapsed();
    println!(
        "l2 512 -> {:.2?} total ({:.2?} per iter), accum {}",
        elapsed,
        elapsed / iterations as u32,
        dist
    );

    // Prevent the compiler from optimizing away the calculations.
    std::hint::black_box(acc);
    std::hint::black_box(dist);
}
