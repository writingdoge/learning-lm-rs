use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
       // todo!("实现从safetensors文件的模型参数加载");
       let get_tensor = |name: &str| -> Tensor<f32> {
            let tv = safetensor.tensor(name).unwrap();
        
            let _data = tv.data();

            assert!(_data.len()%4==0);

            let f32_vec = _data
            .chunks_exact(4) // 每次取 4 个字节
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap(); // 将 4 个字节转换为数组
                f32::from_le_bytes(bytes) // 使用小端字节序转换为 f32
            })
            .collect();


            Tensor::new(f32_vec,&tv.shape().to_vec())

        };


        // let mut res:Vec<Tensor<f32>> = vec![];
        // for i in 0..config.num_hidden_layers{
        //     res.push(get_tensor(format!("model.layers.{}.post_attention_layernorm.weight",i).as_str()));
        // }
        // res

        LLamaParams {
            embedding_table:get_tensor("model.embed_tokens.weight"), //  lm_head.weight
            rms_att_w:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
            .collect(),
            wq:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
            .collect(),
            wk:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
            .collect(),
            wv:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
            .collect(),
            wo:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
            .collect(),
            rms_ffn_w:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)))
            .collect(),
            w_up:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
            .collect(),
            w_gate:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
            .collect(),
            w_down:(0..config.num_hidden_layers)
            .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
            .collect(),
            rms_out_w:get_tensor("model.norm.weight"),
            lm_head:get_tensor("lm_head.weight"),
        }
    }
}
