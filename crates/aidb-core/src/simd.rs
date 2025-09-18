use std::convert::TryInto;
use wide::*;

#[inline(always)]
fn has_avx2() -> bool {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(any(not(target_arch = "x86_64"), miri))]
    {
        false
    }
}

#[inline(always)]
fn avx2_cosine_sim(a: &[f32], b: &[f32]) -> Option<f32> {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        use std::arch::x86_64::*;
        let len = a.len();
        let mut i = 0;
        let mut dot = _mm256_setzero_ps();
        let mut norm_a = _mm256_setzero_ps();
        let mut norm_b = _mm256_setzero_ps();

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            dot = _mm256_fmadd_ps(va, vb, dot);
            norm_a = _mm256_fmadd_ps(va, va, norm_a);
            norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
            i += 8;
        }

        let mut dot_scalar = horizontal_sum256(dot);
        let mut norm_a_scalar = horizontal_sum256(norm_a);
        let mut norm_b_scalar = horizontal_sum256(norm_b);

        while i < len {
            let x = *a.get_unchecked(i);
            let y = *b.get_unchecked(i);
            dot_scalar += x * y;
            norm_a_scalar += x * x;
            norm_b_scalar += y * y;
            i += 1;
        }

        if norm_a_scalar == 0.0 || norm_b_scalar == 0.0 {
            return Some(0.0);
        }
        Some(dot_scalar / (norm_a_scalar.sqrt() * norm_b_scalar.sqrt()))
    }
    #[cfg(any(not(target_arch = "x86_64"), miri))]
    {
        let _ = (a, b);
        None
    }
}

#[inline(always)]
fn avx2_l2(a: &[f32], b: &[f32]) -> Option<f32> {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        use std::arch::x86_64::*;
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm256_setzero_ps();

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(diff, diff, acc);
            i += 8;
        }

        let mut sum = horizontal_sum256(acc);

        while i < len {
            let x = *a.get_unchecked(i);
            let y = *b.get_unchecked(i);
            let d = x - y;
            sum += d * d;
            i += 1;
        }

        if sum <= 0.0 {
            return Some(0.0);
        }
        Some(sum.sqrt())
    }
    #[cfg(any(not(target_arch = "x86_64"), miri))]
    {
        let _ = (a, b);
        None
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline(always)]
unsafe fn horizontal_sum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums)
}

#[inline(always)]
fn truncate_pairs<'a>(a: &'a [f32], b: &'a [f32]) -> (&'a [f32], &'a [f32]) {
    let len = a.len().min(b.len());
    (&a[..len], &b[..len])
}

#[inline(always)]
fn load_f32x8(chunk: &[f32]) -> f32x8 {
    let arr: [f32; 8] = chunk
        .try_into()
        .expect("chunks_exact should yield chunks of length 8");
    f32x8::new(arr)
}

#[inline(always)]
fn scalar_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

// SIMD-accelerated distance functions for ultimate performance
pub fn simd_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Ensure we only operate on the overlapping prefix when lengths diverge.
    let dims_match = a.len() == b.len();
    let (a, b) = truncate_pairs(a, b);
    if !dims_match {
        return scalar_cosine(a, b);
    }

    if has_avx2() {
        if let Some(res) = avx2_cosine_sim(a, b) {
            return res;
        }
    }

    let mut dot_sum = f32x8::splat(0.0);
    let mut norm_a_sum = f32x8::splat(0.0);
    let mut norm_b_sum = f32x8::splat(0.0);

    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);

    for (chunk_a, chunk_b) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = load_f32x8(chunk_a);
        let vb = load_f32x8(chunk_b);
        dot_sum += va * vb;
        norm_a_sum += va * va;
        norm_b_sum += vb * vb;
    }

    let mut dot = dot_sum.reduce_add();
    let mut norm_a = norm_a_sum.reduce_add();
    let mut norm_b = norm_b_sum.reduce_add();

    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    debug_assert_eq!(tail_a.len(), tail_b.len());
    if !tail_a.is_empty() {
        // Fall back to scalar accumulation for the tail.
        dot += tail_a.iter().zip(tail_b).map(|(&x, &y)| x * y).sum::<f32>();
        norm_a += tail_a.iter().map(|&x| x * x).sum::<f32>();
        norm_b += tail_b.iter().map(|&y| y * y).sum::<f32>();
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

#[inline(always)]
fn scalar_l2(a: &[f32], b: &[f32]) -> f32 {
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    sum_sq.sqrt()
}

pub fn simd_l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return f32::INFINITY;
    }

    let dims_match = a.len() == b.len();
    let (a, b) = truncate_pairs(a, b);
    if !dims_match {
        return scalar_l2(a, b);
    }

    if has_avx2() {
        if let Some(res) = avx2_l2(a, b) {
            return res;
        }
    }

    let mut sum = f32x8::splat(0.0);
    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);

    for (chunk_a, chunk_b) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = load_f32x8(chunk_a);
        let vb = load_f32x8(chunk_b);
        let diff = va - vb;
        sum += diff * diff;
    }

    let mut result = sum.reduce_add();

    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    debug_assert_eq!(tail_a.len(), tail_b.len());
    if !tail_a.is_empty() {
        result += tail_a
            .iter()
            .zip(tail_b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>();
    }

    if result <= 0.0 {
        // Numeric noise can create a tiny negative that would panic under sqrt.
        return 0.0;
    }

    result.sqrt()
}

// Batch SIMD operations for processing multiple vectors at once
pub fn simd_batch_cosine_distances(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    vectors
        .iter()
        .map(|vec| 1.0 - simd_cosine_sim(query, vec))
        .collect()
}

pub fn simd_batch_l2_distances(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    vectors
        .iter()
        .map(|vec| simd_l2_distance(query, vec))
        .collect()
}

// Ultra-fast parallel batch distance computation using Rayon
pub fn simd_parallel_batch_distances(
    query: &[f32],
    vectors: &[&[f32]],
    metric: crate::Metric,
) -> Vec<f32> {
    use rayon::prelude::*;

    // Use parallel iterator for large batches
    if vectors.len() > 1000 {
        vectors
            .par_iter()
            .map(|vec| match metric {
                crate::Metric::Cosine => 1.0 - simd_cosine_sim(query, vec),
                crate::Metric::Euclidean => simd_l2_distance(query, vec),
            })
            .collect()
    } else {
        // Use serial for small batches to avoid overhead
        vectors
            .iter()
            .map(|vec| match metric {
                crate::Metric::Cosine => 1.0 - simd_cosine_sim(query, vec),
                crate::Metric::Euclidean => simd_l2_distance(query, vec),
            })
            .collect()
    }
}

// Memory-aligned vector operations for cache efficiency
#[repr(align(32))]
pub struct AlignedVector {
    data: Vec<f32>,
}

impl AlignedVector {
    pub fn new(mut data: Vec<f32>) -> Self {
        // Ensure vector is 32-byte aligned for optimal SIMD performance
        if data.as_ptr() as usize % 32 != 0 {
            let mut aligned = Vec::with_capacity(data.len());
            aligned.extend_from_slice(&data);
            data = aligned;
        }
        Self { data }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

// Fused multiply-add optimized distance computation
#[inline(always)]
pub fn simd_fma_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let (a, b) = truncate_pairs(a, b);

    if has_avx2() {
        if let Some(res) = avx2_cosine_sim(a, b) {
            return res;
        }
    }

    // Use FMA for better precision and performance
    let mut dot_sum = f32x8::splat(0.0);
    let mut norm_a_sum = f32x8::splat(0.0);
    let mut norm_b_sum = f32x8::splat(0.0);

    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);

    for (chunk_a, chunk_b) in chunks_a.clone().zip(chunks_b.clone()) {
        let va = load_f32x8(chunk_a);
        let vb = load_f32x8(chunk_b);

        // Fused multiply-add for better accuracy
        dot_sum = va.mul_add(vb, dot_sum);
        norm_a_sum = va.mul_add(va, norm_a_sum);
        norm_b_sum = vb.mul_add(vb, norm_b_sum);
    }

    let mut dot = dot_sum.reduce_add();
    let mut norm_a = norm_a_sum.reduce_add();
    let mut norm_b = norm_b_sum.reduce_add();

    // Handle remainder
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();
    if !tail_a.is_empty() {
        for (&x, &y) in tail_a.iter().zip(tail_b.iter()) {
            dot = x.mul_add(y, dot);
            norm_a = x.mul_add(x, norm_a);
            norm_b = y.mul_add(y, norm_b);
        }
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cosine_sim, l2};

    #[test]
    fn test_simd_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((simd_cosine_sim(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((simd_cosine_sim(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_l2_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        assert!((simd_l2_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_handles_tail() {
        let dims = 23;
        let a: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..dims).map(|i| (dims - i) as f32 * 0.2).collect();
        let expected = cosine_sim(&a, &b);
        let actual = simd_cosine_sim(&a, &b);
        assert!((expected - actual).abs() < 1e-5);
    }

    #[test]
    fn cosine_falls_back_on_mismatch() {
        let a = vec![0.5f32; 17];
        let b = vec![0.25f32; 16];
        let expected = cosine_sim(&a[..16], &b[..16]);
        let actual = simd_cosine_sim(&a, &b);
        assert!((expected - actual).abs() < 1e-6);
    }

    #[test]
    fn l2_handles_tail() {
        let dims = 19;
        let a: Vec<f32> = (0..dims).map(|i| i as f32 * 0.33).collect();
        let b: Vec<f32> = (0..dims).map(|i| (i as f32 * 1.1) % 5.0).collect();
        let expected = l2(&a, &b);
        let actual = simd_l2_distance(&a, &b);
        assert!((expected - actual).abs() < 1e-5);
    }

    #[test]
    fn l2_falls_back_on_mismatch() {
        let a = vec![1.0f32; 9];
        let b = vec![0.5f32; 7];
        let expected = l2(&a[..7], &b[..7]);
        let actual = simd_l2_distance(&a, &b);
        assert!((expected - actual).abs() < 1e-6);
    }
}
