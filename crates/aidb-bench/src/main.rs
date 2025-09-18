use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use aidb_core::{Metric, VectorIndex};

mod advanced_bench;
use aidb_index_bf::BruteForceIndex;
use aidb_index_hnsw::{HnswIndex, HnswParams};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

fn parse_arg<T: std::str::FromStr>(key: &str, default: T) -> T {
    for a in env::args() {
        if let Some(v) = a.strip_prefix(&format!("--{}=", key)) {
            if let Ok(x) = v.parse::<T>() {
                return x;
            }
        }
    }
    default
}

fn parse_str(key: &str, default: &str) -> String {
    for a in env::args() {
        if let Some(v) = a.strip_prefix(&format!("--{}=", key)) {
            return v.to_string();
        }
    }
    default.to_string()
}

fn gen_unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    let mut n = 0.0f32;
    for i in 0..dim {
        let x = rng.gen::<f32>() - 0.5;
        v[i] = x;
        n += x * x;
    }
    let n = n.sqrt().max(1e-6);
    for i in 0..dim {
        v[i] /= n;
    }
    v
}

fn add_noise(rng: &mut StdRng, v: &[f32], sigma: f32) -> Vec<f32> {
    let mut out = v.to_vec();
    for x in &mut out {
        *x += (rng.gen::<f32>() - 0.5) * sigma;
    }
    // re-normalize
    let mut n = 0.0f32;
    for &x in &out {
        n += x * x;
    }
    let n = n.sqrt().max(1e-6);
    for x in &mut out {
        *x /= n;
    }
    out
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[k]
}

#[derive(Serialize)]
struct Report {
    mode: String,
    index: String,
    dim: usize,
    n: usize,
    q: usize,
    topk: usize,
    metric: String,
    ingest_sec: f64,
    ingest_kps: f64,
    search_sec: f64,
    qps: f64,
    lat_avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn bench_index<I: VectorIndex>(
    mut index: I,
    dim: usize,
    n: usize,
    q: usize,
    top_k: usize,
    metric: Metric,
    seed: u64,
) -> Report {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut base = Vec::with_capacity(n);
    for _ in 0..n {
        base.push(gen_unit_vec(&mut rng, dim));
    }

    // insert
    let t0 = Instant::now();
    for (i, v) in base.iter().enumerate() {
        let id = uuid::Uuid::from_u128(i as u128);
        index.add(id, v.clone(), None);
    }
    let t_ins = t0.elapsed().as_secs_f64();

    // queries near existing points
    let mut latencies = Vec::with_capacity(q);
    let t1 = Instant::now();
    for _ in 0..q {
        let base_idx = rng.gen_range(0..n);
        let query = add_noise(&mut rng, &base[base_idx], 0.1);
        let qs = Instant::now();
        let _res = index.search(&query, top_k, metric, None);
        let dt = qs.elapsed().as_secs_f64() * 1e3; // ms
        latencies.push(dt);
    }
    let t_search = t1.elapsed().as_secs_f64();

    let p50 = percentile(latencies.clone(), 0.50);
    let p95 = percentile(latencies.clone(), 0.95);
    let p99 = percentile(latencies.clone(), 0.99);
    let avg = latencies.iter().copied().sum::<f64>() / (latencies.len().max(1) as f64);

    Report {
        mode: "index".into(),
        index: "".into(),
        dim,
        n,
        q,
        topk: top_k,
        metric: match metric {
            Metric::Cosine => "cosine".into(),
            Metric::Euclidean => "euclidean".into(),
        },
        ingest_sec: t_ins,
        ingest_kps: (n as f64 / t_ins) / 1e3,
        search_sec: t_search,
        qps: q as f64 / t_search,
        lat_avg_ms: avg,
        p50_ms: p50,
        p95_ms: p95,
        p99_ms: p99,
    }
}

#[tokio::main]
async fn main() {
    let dim = parse_arg("dim", 384usize);
    let n = parse_arg("n", 50_000usize);
    let q = parse_arg("q", 200usize);
    let top_k = parse_arg("topk", 10usize);
    let seed = parse_arg("seed", 42u64);
    let metric_s = parse_str("metric", "cosine");
    let metric = match metric_s.as_str() {
        "euclidean" => Metric::Euclidean,
        _ => Metric::Cosine,
    };

    let mode = parse_str("mode", "index");
    let report_fmt = parse_str("report", ""); // "json" | "csv" | ""
    let out_path = parse_str("out", ""); // file path or empty for stdout

    let report = if mode == "advanced" {
        // Run advanced benchmarks
        println!("Running advanced AIDB benchmarks...");

        let mut all_results = Vec::new();

        println!("Testing SIMD optimizations...");
        all_results.extend(advanced_bench::run_simd_benchmarks());

        println!("Testing batch processing...");
        all_results.extend(advanced_bench::run_batch_processing_benchmarks());

        println!("Testing memory efficiency...");
        all_results.extend(advanced_bench::run_memory_efficiency_tests());

        let report_text = advanced_bench::generate_benchmark_report(&all_results);
        println!("{}", report_text);

        return; // Exit early for advanced benchmarks
    } else if mode == "http" {
        http_bench(dim, n, q, top_k, metric, seed).await
    } else {
        let index_s = parse_str("index", "hnsw");
        match index_s.as_str() {
            "bruteforce" => {
                let index = BruteForceIndex::new(dim);
                let mut r = bench_index(index, dim, n, q, top_k, metric, seed);
                r.index = "bruteforce".into();
                r
            }
            "hnsw" => {
                let m = parse_arg("m", 16usize);
                let efc = parse_arg("efc", 200usize);
                let efs = parse_arg("efs", 50usize);
                let index = HnswIndex::try_new(
                    dim,
                    metric,
                    HnswParams {
                        m,
                        ef_construction: efc,
                        ef_search: efs,
                    },
                )
                .expect("valid HNSW params");
                let mut r = bench_index(index, dim, n, q, top_k, metric, seed);
                r.index = format!("hnsw(m={},efc={},efs={})", m, efc, efs);
                r
            }
            other => panic!("unknown --index={}", other),
        }
    };

    output_report(&report, &report_fmt, &out_path);
}

fn output_report(r: &Report, fmt: &str, path: &str) {
    if fmt == "json" {
        let s = serde_json::to_string_pretty(r).unwrap();
        if path.is_empty() {
            println!("{}", s);
        } else {
            let mut f = File::create(path).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        }
    } else if fmt == "csv" {
        let header = "mode,index,dim,n,q,topk,metric,ingest_sec,ingest_kps,search_sec,qps,lat_avg_ms,p50_ms,p95_ms,p99_ms";
        let line = format!(
            "{},{},{},{},{},{},{},{:.6},{:.3},{:.6},{:.3},{:.3},{:.3},{:.3},{:.3}",
            r.mode,
            r.index,
            r.dim,
            r.n,
            r.q,
            r.topk,
            r.metric,
            r.ingest_sec,
            r.ingest_kps,
            r.search_sec,
            r.qps,
            r.lat_avg_ms,
            r.p50_ms,
            r.p95_ms,
            r.p99_ms
        );
        if path.is_empty() {
            println!("{}\n{}", header, line);
        } else {
            let mut f = File::create(path).unwrap();
            writeln!(f, "{}", header).unwrap();
            writeln!(f, "{}", line).unwrap();
        }
    } else {
        println!(
            "mode={} index={} dim={} n={} q={} topk={} metric={}",
            r.mode, r.index, r.dim, r.n, r.q, r.topk, r.metric
        );
        println!(
            "ingest: {:.3}s total, {:.1} K pts/s",
            r.ingest_sec, r.ingest_kps
        );
        println!("search: {:.3}s total, {:.1} QPS", r.search_sec, r.qps);
        println!(
            "latency(ms): avg={:.3} p50={:.3} p95={:.3} p99={:.3}",
            r.lat_avg_ms, r.p50_ms, r.p95_ms, r.p99_ms
        );
    }
}

async fn http_bench(
    dim: usize,
    n: usize,
    q: usize,
    top_k: usize,
    metric: Metric,
    seed: u64,
) -> Report {
    let base = parse_str("base", "http://127.0.0.1:8080");
    let name = parse_str("col", "bench");
    let index_s = parse_str("index", "hnsw");
    let m = parse_arg("m", 16usize);
    let efc = parse_arg("efc", 200usize);
    let efs = parse_arg("efs", 50usize);
    let batch = parse_arg("batch", 1000usize);
    let mut rng = StdRng::seed_from_u64(seed);
    let client = reqwest::Client::new();

    // create collection
    let metric_s = match metric {
        Metric::Cosine => "cosine",
        Metric::Euclidean => "euclidean",
    };
    let mut body = serde_json::json!({ "name": name, "dim": dim, "metric": metric_s, "wal_dir": "./data", "index": index_s });
    if index_s == "hnsw" {
        body["hnsw"] = serde_json::json!({"m": m, "ef_construction": efc, "ef_search": efs});
    }
    client
        .post(format!("{}/collections", base))
        .json(&body)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();

    // generate dataset
    let mut base_vecs = Vec::with_capacity(n);
    for _ in 0..n {
        base_vecs.push(gen_unit_vec(&mut rng, dim));
    }

    // ingest in batches
    let t0 = Instant::now();
    for chunk in base_vecs.chunks(batch) {
        let points: Vec<_> = chunk
            .iter()
            .map(|v| serde_json::json!({"vector": v, "payload": null}))
            .collect();
        let body = serde_json::json!({"points": points});
        client
            .post(format!("{}/collections/{}/points:batch", base, name))
            .json(&body)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
    }
    let t_ins = t0.elapsed().as_secs_f64();

    // queries
    let mut latencies = Vec::with_capacity(q);
    let t1 = Instant::now();
    for _ in 0..q {
        let base_idx = rng.gen_range(0..n);
        let query = add_noise(&mut rng, &base_vecs[base_idx], 0.1);
        let qs = Instant::now();
        let body = serde_json::json!({"vector": query, "top_k": top_k});
        let _ = client
            .post(format!("{}/collections/{}/search", base, name))
            .json(&body)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
        let dt = qs.elapsed().as_secs_f64() * 1e3; // ms
        latencies.push(dt);
    }
    let t_search = t1.elapsed().as_secs_f64();

    let p50 = percentile(latencies.clone(), 0.50);
    let p95 = percentile(latencies.clone(), 0.95);
    let p99 = percentile(latencies.clone(), 0.99);
    let avg = latencies.iter().copied().sum::<f64>() / (latencies.len().max(1) as f64);

    Report {
        mode: "http".into(),
        index: index_s,
        dim,
        n,
        q,
        topk: top_k,
        metric: metric_s.into(),
        ingest_sec: t_ins,
        ingest_kps: (n as f64 / t_ins) / 1e3,
        search_sec: t_search,
        qps: q as f64 / t_search,
        lat_avg_ms: avg,
        p50_ms: p50,
        p95_ms: p95,
        p99_ms: p99,
    }
}
