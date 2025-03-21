use crate::tensor::Tensor;
use half::f16;
use rayon::prelude::*;

// 通用 Float 计算 trait，支持混合精度计算
pub trait FloatElement: Copy + Clone + Default + Send + Sync {
    fn to_f32(self) -> f32;
    fn from_f32(val: f32) -> Self;
    fn from_f16(val: f16) -> Self;
}

impl FloatElement for f32 {
    fn to_f32(self) -> f32 { self }
    fn from_f32(val: f32) -> Self { val }
    fn from_f16(val: f16) -> Self{val.to_f32()}
}

impl FloatElement for half::f16 {
    fn to_f32(self) -> f32 { self.to_f32() }
    fn from_f32(val: f32) -> Self { half::f16::from_f32(val) }
    fn from_f16(val: f16) -> Self { val }
}


// get (row) vectors from a 2D table given a list of indices
// table 可以是f32，f16
pub fn gather<T: FloatElement>(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<T>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert_eq!(table_shape.len(), 2);
    let dim = table_shape[1];
    assert_eq!(y.size(), length * dim);
    // y: length,dim
    // table: table_len,dim
    // indice: length (x<table_len)
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        for j in 0..dim {
            unsafe{
            y.data_mut()[i * dim + j] = src[j].to_f32();
            }
        }
        // let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        // dst.copy_from_slice(src);
    }
}

pub fn gather_parallel<T: FloatElement>(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<T>) {
        let length = indices.size();
        let table_shape = table.shape();
        assert_eq!(table_shape.len(), 2);
        let dim = table_shape[1];
        assert_eq!(y.size(), length * dim);
    
        let _table = table.data();
        let _indices = indices.data();
    
        //核心：按 dim 分块，避免外层捕获 y
        unsafe {
            y.data_mut()
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(i, dst)| {
                    let idx = _indices[i] as usize;
                    let src = &_table[idx * dim..(idx + 1) * dim];
                    for j in 0..dim {
                        dst[j] = src[j].to_f32();
                    }
                });
        }
    }


// RoPE: Rotary Positional Embedding
// y 可以是f32，f16
pub fn rope<T: FloatElement>(y: &mut Tensor<T>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert_eq!(shape.len(), 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = (pos as f32) / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                let a_val = a.to_f32();
                let b_val = b.to_f32();
                data[tok * n_heads * d + head * d + i] = T::from_f32(a_val * cos - b_val * sin);
                data[tok * n_heads * d + head * d + i + d / 2] = T::from_f32(b_val * cos + a_val * sin);
            }
        }
    }
}

pub fn rope_parallel<T: FloatElement>(y: &mut Tensor<T>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert_eq!(shape.len(), 3);
    // let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];

    let chunk_size = n_heads * d;
    let data = unsafe { y.data_mut() };

    // 按 token 维度切块
    data.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(tok, chunk)| {
            let pos = start_pos + tok;
            for head in 0..n_heads {
                for i in 0..d / 2 {
                    let a = chunk[head * d + i];
                    let b = chunk[head * d + i + d / 2];
                    let freq = (pos as f32) / theta.powf((i * 2) as f32 / d as f32);
                    let (sin, cos) = freq.sin_cos();
                    let a_val = a.to_f32();
                    let b_val = b.to_f32();
                    chunk[head * d + i] = T::from_f32(a_val * cos - b_val * sin);
                    chunk[head * d + i + d / 2] = T::from_f32(b_val * cos + a_val * sin);
                }
            }
        });
}

// pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
//     let shape = y.shape();
//     assert!(shape.len() == 3);
//     let seq_len = shape[0];
//     let n_heads = shape[1];
//     let d = shape[2];
//     let data = unsafe { y.data_mut() };
//     for tok in 0..seq_len {
//         let pos = start_pos + tok;
//         for head in 0..n_heads {
//             for i in 0..d / 2 {
//                 let a = data[tok * n_heads * d + head * d + i];
//                 let b = data[tok * n_heads * d + head * d + i + d / 2];
//                 let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
//                 let (sin, cos) = freq.sin_cos();
//                 data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
//                 data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
//             }
//         }
//     }
// }

// 混合精度 RMSNorm 
// w 可以是f32,f16
pub fn rms_norm<T: FloatElement>(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<T>, epsilon: f32) {
    // println!("rms_norm");
    // println!("y: {:?} x: {:?} w: {:?}", y.shape(), x.shape(), w.shape());

    // 确保x、y形状相等，w是长度为n的向量，x、y最后一维长度为n
    assert_eq!(y.shape(), x.shape());
    assert_eq!(w.shape().len(), 1);
    assert_eq!(*x.shape().last().unwrap(), w.size());

    let n = x.shape().last().unwrap();
    let mut _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();

    let mut  ny:Vec < Vec <f32> > = vec![];

    for xi in _x.chunks(*n) {
        // let mut sum = xi.iter().fold(0.,|acc:f32,x|acc+x*x);
        let mut sum: f32 = xi.iter().map(|&x| x*x).sum();
        sum /= *n as f32;
        sum += epsilon;
        let norm = sum.sqrt();

        let result: Vec<f32> = xi.iter().zip(_w.iter()).map(|(&xij, &wj)| {
            let w_val = wj.to_f32();
            xij * w_val / norm
        }).collect();

        ny.push(result);
    }

    let flat_slice: Vec<f32> = ny.iter().flat_map(|vec| vec.iter()).cloned().collect();

    _y.copy_from_slice(&flat_slice);
}



// 多线程并行优化
pub fn rms_norm_parallel<T: FloatElement>(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<T>, epsilon: f32) {
    assert_eq!(y.shape(), x.shape());
    assert_eq!(w.shape().len(), 1);
    assert_eq!(*x.shape().last().unwrap(), w.size());

    let n = *x.shape().last().unwrap();
    let _x = x.data();
    let _w = w.data();
    let _y = unsafe { y.data_mut() };

    _x.par_chunks(n)
        .zip(_y.par_chunks_mut(n))
        .for_each(|(xi, yi)| {
            let mut sum: f32 = xi.iter().map(|&x| x * x).sum();
            sum /= n as f32;
            sum += epsilon;
            let norm = sum.sqrt();

            for (i, (&xij, &wj)) in xi.iter().zip(_w.iter()).enumerate() {
                let w_val = wj.to_f32();
                yi[i] = xij * w_val / norm;
            }
        });
}


// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn masked_softmax_parallel(y: &mut Tensor<f32>) {
        let ndim = y.shape().len();
        assert!(ndim >= 2);
        let seq_len = y.shape()[ndim - 2];
        let total_seq_len = y.shape()[ndim - 1];
    
        let data = unsafe { y.data_mut() };
        data.par_chunks_mut(seq_len * total_seq_len)
        .for_each(|chunk| {
        for i in 0..seq_len {
            let offset = i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let row = &mut chunk[offset..offset + total_seq_len];
            let (active, masked) = row.split_at_mut(boundary);

            let max = active.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = active.iter_mut().map(|v| {
                *v = (*v - max).exp();
                *v
            }).sum();

            active.iter_mut().for_each(|v| *v /= sum);
            masked.iter_mut().for_each(|v| *v = 0.0);
        }
    });
    }


// y = silu(x) * y
// hint: this is an element-wise operation
// y,x可以全是f16
pub fn swiglu<T: FloatElement>(y: &mut Tensor<T>, x: &Tensor<T>) {
    let len = y.size();
    assert!(len == x.size());

    let e = std::f64::consts::E;

    let mut _y = unsafe { y.data_mut() };
    let _x = x.data();
    // 
    let sigmoid_x:Vec<f32> = _x.iter().map(|a|1./(1.+e.powf(-a.to_f32() as f64)) as f32).collect();

    let silu_x: Vec<f32> = _x.iter().zip(sigmoid_x.iter()).map(|(s1, s2)| s1.to_f32() * s2).collect();
    
    let result: Vec<T> = _y.iter()
    .zip(silu_x.iter())
    .map(|(s1, s2)| T::from_f32(s1.to_f32() * s2))
    .collect();

    _y.copy_from_slice(&result); 
    // _y = _y.iter().zip(silu_x.iter()).map(|(s1, s2)| s1 * s2).collect();


    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
}

pub fn swiglu_parallel<T: FloatElement>(y: &mut Tensor<T>, x: &Tensor<T>) {
    let len = y.size();
    assert_eq!(len, x.size());

    let e = std::f64::consts::E;
    let _x = x.data();
    let _y = unsafe { y.data_mut() };

    // 并行计算 sigmoid 和 silu 一步完成
    _y.par_iter_mut().zip(_x.par_iter()).for_each(|(yi, xi)| {
        let x_val = xi.to_f32();
        let sigmoid = 1.0 / (1.0 + e.powf(-x_val as f64)) as f32;
        let silu = x_val * sigmoid;
        *yi = T::from_f32(yi.to_f32() * silu);
    });
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
// 你可以默认输入输出都是二维矩阵，即 $`A`$ 形状为 $`m×k`$，$`B`$ 形状为 $`n×k`$，$`C`$ 形状为 $`m×n`$，
// 可以不用考虑广播的情况。

// c,b是f16;b是f162种情况
pub fn matmul_transb<T: FloatElement, S: FloatElement,  R: FloatElement>(c: &mut Tensor<T>, beta: f32, a: &Tensor<R>, b: &Tensor<S>, alpha: f32) {
    // A: m*k  B: n*k C:m*n
    assert_eq!(a.shape()[1],b.shape()[1]);
    assert_eq!(a.shape()[0],c.shape()[0]);
    assert_eq!(b.shape()[0],c.shape()[1]);

    let k = a.shape()[1];
    // let m = a.shape()[0];
    let n = b.shape()[0];

    for i in 0..a.size()/k{
        for j in 0..b.size()/k{
            // println!("{i} {j}");
            let _a = a.data();
            let _b = b.data();
            // _a[i*k..][..i*k+k]

            // _b[j*k..][..j*k+k]

            let sum :f32= _a[i*k..][..k].iter()
            .zip(_b[j*k..][..k].iter()).map(|(a,b)| a.to_f32()*b.to_f32()).sum();

            unsafe{
                let _c = c.data_mut();

                _c[i*n+j] = T::from_f32(beta*(_c[i*n+j].to_f32()) + alpha*sum) ;

            }
        }
    }
}

//简单按行并行
pub fn matmul_transb_parallel<T: FloatElement, S: FloatElement, R: FloatElement>(
    c: &mut Tensor<T>,
    beta: f32,
    a: &Tensor<R>,
    b: &Tensor<S>,
    alpha: f32,
) {
    assert_eq!(a.shape()[1], b.shape()[1]); // k
    assert_eq!(a.shape()[0], c.shape()[0]); // m
    assert_eq!(b.shape()[0], c.shape()[1]); // n

    let k = a.shape()[1];
    let m = a.shape()[0];
    let n = b.shape()[0];

    let _a = a.data();
    let _b = b.data();
    let _c = unsafe { c.data_mut() };

    // 1块n个
    _c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        for j in 0..n {
            let sum: f32 = _a[i * k..i * k + k]
                .iter()
                .zip(&_b[j * k..j * k + k])
                .map(|(a, b)| a.to_f32() * b.to_f32())
                .sum();

            c_row[j] = T::from_f32(beta * c_row[j].to_f32() + alpha * sum);
        }
    });
}



// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu_parallel(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y   = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm_parallel(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb_parallel(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}



use crate::benchmark::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn test_rms_norm_opt(){ 
    let shape = vec![128, 4096];
    let seed = 42;

    let mut y_origin = Tensor::<f32>::default(&shape);
    let mut y_parallel = Tensor::<f32>::default(&shape);
    let x = random_tensor_f32(&shape, seed);
    let w = random_tensor_f32(&[shape[1]], seed + 1);

    rms_norm(&mut y_origin, &x, &w, 1e-6);
    rms_norm_parallel(&mut y_parallel, &x, &w, 1e-6);

    let diff: f32 = y_origin
    .data()
    .iter()
    .zip(y_parallel.data().iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-4);

    let origin = benchmark(|| {
        rms_norm(&mut y_origin, &x, &w, 1e-6);
    }, 5, 20);

    let opt = benchmark(|| {
        rms_norm_parallel(&mut y_parallel, &x, &w, 1e-6);
    }, 5, 20);

    println!("原版平均时间: {:.3}ms | 优化后平均时间: {:.3}ms", origin, opt);

}
// 原版平均时间: 45.493ms | 优化后平均时间: 5.680ms

#[test]
fn test_rms_norm_opt_fp16(){ 
    let shape = vec![128, 4096];
    let seed = 42;

    let mut y_origin = Tensor::<f32>::default(&shape);
    let mut y_parallel = Tensor::<f32>::default(&shape);
    let x = random_tensor_f32(&shape, seed);
    let w = random_tensor_f16(&[shape[1]], seed + 1); 

    rms_norm(&mut y_origin, &x, &w, 1e-6);
    rms_norm_parallel(&mut y_parallel, &x, &w, 1e-6);

    let diff: f32 = y_origin
    .data()
    .iter()
    .zip(y_parallel.data().iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-2);

    let origin = benchmark(|| {
        rms_norm(&mut y_origin, &x, &w, 1e-6);
    }, 5, 20);

    let opt = benchmark(|| {
        rms_norm_parallel(&mut y_parallel, &x, &w, 1e-6);
    }, 5, 20);

    println!("FP16原版平均时间: {:.3}ms | FP16优化后平均时间: {:.3}ms", origin, opt);
}
// FP16原版平均时间: 117.596ms | FP16优化后平均时间: 29.058ms

#[test]
fn test_gather_opt(){ 
    // y: length,dim
    // table: table_len,dim
    // indice: length (x<table_len)
    let shape_y = vec![128, 4096];
    let shape_table = vec![8192,4096];
    let shape_indice = vec![128];
    let seed = 42;

    fn random_tensor_u32(shape: &[usize], seed: u64) -> Tensor<u32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let size: usize = shape.iter().product();
        let data:Vec<u32> = (0..size).map(|_| rng.gen_range(0..8191)).collect();
        Tensor::new(data, &shape.to_vec())
    }

    let mut y_origin = Tensor::<f32>::default(&shape_y);
    let mut y_parallel = Tensor::<f32>::default(&shape_y);
    let table = random_tensor_f32(&shape_table, seed);
    let indices = random_tensor_u32(&shape_indice,seed);

    gather(&mut y_origin,&indices,&table);
    gather_parallel(&mut y_parallel,&indices,&table);

    let diff: f32 = y_origin
    .data()
    .iter()
    .zip(y_parallel.data().iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-4);

    let origin = benchmark(|| {
        gather(&mut y_origin,&indices,&table);
    }, 5, 20);

    let opt = benchmark(|| {
        gather_parallel(&mut y_parallel,&indices,&table);
    }, 5, 20);

    println!("原版平均时间: {:.3}ms | 优化后平均时间: {:.3}ms", origin, opt);

}
// 原版平均时间: 20.421ms | 优化后平均时间: 2.188ms

#[test]
fn test_gather_opt_fp16(){ 
    let shape_y = vec![128, 4096];
    let shape_table = vec![8192,4096];
    let shape_indice = vec![128];
    let seed = 42;

    fn random_tensor_u32(shape: &[usize], seed: u64) -> Tensor<u32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let size: usize = shape.iter().product();
        let data:Vec<u32> = (0..size).map(|_| rng.gen_range(0..8191)).collect();
        Tensor::new(data, &shape.to_vec())
    }

    let mut y_origin = Tensor::<f32>::default(&shape_y);
    let mut y_parallel = Tensor::<f32>::default(&shape_y);
    let table = random_tensor_f16(&shape_table, seed); // 使用f16版本的table
    let indices = random_tensor_u32(&shape_indice,seed);

    gather(&mut y_origin,&indices,&table);
    gather_parallel(&mut y_parallel,&indices,&table);

    let diff: f32 = y_origin
    .data()
    .iter()
    .zip(y_parallel.data().iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-2);

    let origin = benchmark(|| {
        gather(&mut y_origin,&indices,&table);
    }, 5, 20);

    let opt = benchmark(|| {
        gather_parallel(&mut y_parallel,&indices,&table);
    }, 5, 20);

    println!("FP16原版平均时间: {:.3}ms | FP16优化后平均时间: {:.3}ms", origin, opt);
}
// FP16原版平均时间: 54.006ms | FP16优化后平均时间: 14.936ms

#[test]
fn test_masked_softmax_opt(){ 
    let shape = vec![1028, 4096];
    let seed = 42;

    let mut y_origin = random_tensor_f32(&shape, seed);

    // let mut softmax_out_origin = y.clone();
    let mut y_parallel = y_origin.clone();

    masked_softmax(&mut y_origin);
    masked_softmax(&mut y_parallel);

    let diff: f32 = y_origin
    .data()
    .iter()
    .zip(y_parallel.data().iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-4);

    let origin = benchmark(|| {
        masked_softmax(&mut y_origin);
    }, 5, 20);

    let opt = benchmark(|| {
        masked_softmax_parallel(&mut y_parallel);
    }, 5, 20);

    println!("原版平均时间: {:.3}ms | 优化后平均时间: {:.3}ms", origin, opt);
}
// 原版平均时间: 272.774ms | 优化后平均时间: 194.225ms


#[test]
fn test_swiglu_opt(){ 
    let shape = vec![8192];
    let seed = 42;

    let mut y_origin = random_tensor_f32(&shape, seed);
    let mut y_parallel = y_origin.clone();
    let x = random_tensor_f32(&shape, seed + 1);

    swiglu(&mut y_origin, &x);
    swiglu(&mut y_parallel, &x);

    let diff: f32 = y_origin
    .data()
    .iter()
    .zip(y_parallel.data().iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-4);

    let origin = benchmark(|| {
       swiglu(&mut y_origin, &x);
    }, 5, 20);

    let opt = benchmark(|| {
        swiglu_parallel(&mut y_parallel, &x);
    }, 5, 20);

    println!("原版平均时间: {:.3}ms | 优化后平均时间: {:.3}ms", origin, opt);
}
// 原版平均时间: 0.880ms | 优化后平均时间: 0.587ms

#[test]
fn test_swiglu_opt_fp16(){ 
    let shape = vec![8192];
    let seed = 42;

    let mut y_origin = random_tensor_f16(&shape, seed);
    let mut y_parallel = y_origin.clone();
    let x = random_tensor_f16(&shape, seed + 1);

    swiglu(&mut y_origin, &x);
    swiglu_parallel(&mut y_parallel, &x);

    let diff: f32 = y_origin
    .data()
    .iter()
    .zip(y_parallel.data().iter())
    .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-2);

    let origin = benchmark(|| {
       swiglu(&mut y_origin, &x);
    }, 5, 20);

    let opt = benchmark(|| {
        swiglu_parallel(&mut y_parallel, &x);
    }, 5, 20);

    println!("FP16原版平均时间: {:.3}ms | FP16优化后平均时间: {:.3}ms", origin, opt);
}
// FP16原版平均时间: 3.351ms | FP16优化后平均时间: 1.170ms

#[test]
fn test_matmul_transb_opt(){ 
    let shape_a = vec![1000,1000];
    let shape_b = vec![1000,1000];
    let shape_c = vec![1000,1000];
    let seed = 42;

    let a = random_tensor_f32(&shape_a, seed);
    let b = random_tensor_f32(&shape_b, seed + 1);
    let mut c_origin = random_tensor_f32(&shape_c, seed + 2);
    let mut c_parallel = c_origin.clone();
    let beta = 1.;
    let alpha = 1.;

    let diff: f32 = c_origin
    .data()
    .iter()
    .zip(c_parallel.data().iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-4);

    let origin = benchmark(|| {
        matmul_transb(&mut c_origin, beta,&a,&b,alpha);
    }, 1, 3);

    let opt = benchmark(|| {
        matmul_transb_parallel(&mut c_parallel, beta,&a,&b,alpha);
    }, 1, 3);

    println!("原版平均时间: {:.3}ms | 优化后平均时间: {:.3}ms", origin, opt);
}
// 原版平均时间: 9830.634ms | 优化后平均时间: 3775.616ms
#[test]
fn test_matmul_transb_opt_fp16(){ 
    let shape_a = vec![1000,1000];
    let shape_b = vec![1000,1000];
    let shape_c = vec![1000,1000];
    let seed = 42;

    let a = random_tensor_f32(&shape_a, seed);
    let b = random_tensor_f16(&shape_b, seed + 1); // b使用f16
    let mut c_origin = random_tensor_f16(&shape_c, seed + 2); // c使用f16
    let mut c_parallel = c_origin.clone();
    let beta = 1.;
    let alpha = 1.;

    matmul_transb(&mut c_origin, beta, &a, &b, alpha);
    matmul_transb_parallel(&mut c_parallel, beta, &a, &b, alpha);

    let diff: f32 = c_origin
    .data()
    .iter()
    .zip(c_parallel.data().iter())
    .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
    .fold(0.0, |acc, x| acc.max(x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-2);

    let origin = benchmark(|| {
        matmul_transb(&mut c_origin, beta, &a, &b, alpha);
    }, 1, 3);

    let opt = benchmark(|| {
        matmul_transb_parallel(&mut c_parallel, beta, &a, &b, alpha);
    }, 1, 3);

    println!("FP16原版平均时间: {:.3}ms | FP16优化后平均时间: {:.3}ms", origin, opt);
}
// FP16原版平均时间: 142974.160ms | FP16优化后平均时间: 62983.698ms