use std::{usize, vec};

use crate::tensor::Tensor;
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<T: Default + Copy> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize){
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }


    pub fn create_snapshot(&self) -> Self {
        KVCache {
            k_cache: self.k_cache.clone(),
            v_cache: self.v_cache.clone(),
            length: self.length,
            max_seq_len: self.max_seq_len,
            dim: self.dim,
        }
    }

    pub fn load_snapshot(&mut self, snapshot: &KVCache<T>) {
        // 覆盖当前k_cache, v_cache, length
        self.k_cache = snapshot.k_cache.clone();
        self.v_cache = snapshot.v_cache.clone();
        self.length = snapshot.length;
        self.dim = snapshot.dim;
        self.max_seq_len = snapshot.max_seq_len;
    }
}

