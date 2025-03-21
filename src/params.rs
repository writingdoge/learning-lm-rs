use rayon::prelude::*;
use half::f16;
use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;

pub trait FromF32Tensor: Sized + Send + Sync {
    fn from_f32(val: f32) -> Self;
}

impl FromF32Tensor for f32 {
    fn from_f32(val: f32) -> Self { val }
}

impl FromF32Tensor for f16 {
    fn from_f32(val: f32) -> Self { f16::from_f32(val) }
}

pub struct LLamaParams<T> {
    pub embedding_table: Tensor<T>,
    pub rms_att_w: Vec<Tensor<T>>,
    pub wq: Vec<Tensor<T>>,
    pub wk: Vec<Tensor<T>>,
    pub wv: Vec<Tensor<T>>,
    pub wo: Vec<Tensor<T>>,
    pub rms_ffn_w: Vec<Tensor<T>>,
    pub w_up: Vec<Tensor<T>>,
    pub w_gate: Vec<Tensor<T>>,
    pub w_down: Vec<Tensor<T>>,
    pub rms_out_w: Tensor<T>,
    pub lm_head: Tensor<T>,
}

impl<T: FromF32Tensor + Send + Sync + Copy + Default> LLamaParams<T> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<T> {
            let tv = safetensor.tensor(name).unwrap();
            let _data = tv.data();
            assert!(_data.len() % 4 == 0);
            let vec = _data.par_chunks(4)  // 并行化
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    let val_32 = f32::from_le_bytes(bytes);
                    T::from_f32(val_32)
                })
                .collect();
            Tensor::new(vec, &tv.shape().to_vec())
        };

        LLamaParams {
            embedding_table: get_tensor("model.embed_tokens.weight"),

            // 全换成 rayon 并行层加载！
            rms_att_w: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
                .collect(),
            wq: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
                .collect(),
            wk: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
                .collect(),
            wv: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
                .collect(),
            wo: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
                .collect(),
            rms_ffn_w: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)))
                .collect(),
            w_up: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
                .collect(),
            w_gate: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
                .collect(),
            w_down: (0..config.num_hidden_layers).into_par_iter()
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
                .collect(),

            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
