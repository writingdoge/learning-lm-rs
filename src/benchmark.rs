// benchmark.rs 模块化版本

use std::time::Instant;
use rand::{Rng, SeedableRng, rngs::StdRng};
use crate::tensor::Tensor;
use crate::operators::*;


pub fn random_tensor_f32(shape: &[usize], seed: u64) -> Tensor<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let size: usize = shape.iter().product();
    let data = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
    Tensor::new(data, &shape.to_vec())
}

pub fn random_tensor_f16(shape: &[usize], seed: u64) -> Tensor<half::f16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let size: usize = shape.iter().product();
    let data = (0..size).map(|_| half::f16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    Tensor::new(data, &shape.to_vec())
}


pub fn benchmark<F: FnMut()>(
    mut op: F,
    warmup: usize,
    runs: usize,
) -> f64 {
    use std::time::Instant;
    // Warmup
    for _ in 0..warmup {
        op();
    }

    let mut total_time = 0.0;
    for _ in 0..runs {
        let start = Instant::now();
        op();
        total_time += start.elapsed().as_secs_f64() * 1000.0;
    }
    total_time / runs as f64
}

