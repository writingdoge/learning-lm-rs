use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        // println!("{safetensor:?}");
        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h; // 

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        // println!("residual {:?}",residual.data()[0]);
        // println!("{}",self.n_layers);
        for layer in 0..self.n_layers {
            // println!("{} ",layer);
            // println!("residual {:?}",residual.data()[0]);
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            //todo!("self_attention(...)");
            // println!("q {:?}",q.data()[0]);

            //
            self_attention(& mut hidden_states,& mut att_scores,
            q,full_k,full_v,
            self.n_kv_h,n_groups,seq_len,total_seq_len,self.dqkv);


           // todo!("down_proj matmul and add residual");

             // (seq, n_kv_h * n_groups * dqkv)
            // out = attn_V @ O_weight.T     (q_head,len,dim)   wo:(hidden_size, n_heads * head_size)
    
            // C = beta * C + alpha * A @ B^T
            // residual = out + residual
            let wo = & self.params.wo[layer];
            OP::matmul_transb(&mut residual, 1., & hidden_states,wo , 1.);

            // println!("{:?}",residual.shape());
            // println!("after self-attn : {:?}",residual.data()[0]);
           // todo!("mlp(...)");
           mlp(&mut residual,&mut hidden_states,
            &mut gate_buf,&mut up_buf,&self.params.w_up[layer],
        &self.params.w_down[layer],
    &self.params.w_gate[layer],&self.params.rms_ffn_w[layer],self.eps);
        }

        // println!("after mlp : {:?}",residual.data()[0]);

        // println!("yeah everything else is OK");
        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1,self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![1,self.d]);
        // seq_len, self.d

        // println!("last norm: input {:?}",residual.data()[0]);
        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );
        // println!("output {:?}",hidden_states.data()[0]);

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits // 概率分布
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kv: &mut KVCache<f32>,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();
        // let mut kv = self.new_cache();

        let mut inputs = token_ids.to_vec();
        
        //todo!("实现文本生成");
        while result.len()  < max_len{
            let input_tensor = Tensor::new(inputs.clone(),&vec![inputs.len()]);
            let res = self.forward(&input_tensor,kv);
            let next_token = OP::random_sample(&res,top_p,top_k,temperature);
            if next_token == self.eos_token_id{
                break;
            }
            result.push(next_token);
            inputs.clear();
            inputs.push(next_token);

            }

        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq) 
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // (seq, n_kv_h * n_groups * dqkv) (total_seq, n_kv_h * dqkv).T
    // (n_kv_h, n_groups, seq, total_seq)
    // 手动进行索引和向量乘法

    // score = Q @ K.T / sqrt(dim) 
    // 对于每个独立的“头”都得到一个 (seq_len, total_seq_len) 的权重矩阵
    // 对每个seq: (seq,n_kv_h * n_groups * dqkv) (n_kv_h * dqkv,total_seq)
    // @ save to -> (seq,total_seq)


    let _qdata = q.data();
    let _kdata=k.data();
    // i j k i*(n_kv_h * n_groups * dqkv)+j*(n_groups * dqkv) + k*(dqkv)

    // (seq, n_kv_h * n_groups * dqkv)
    // (total_seq, n_kv_h * dqkv)

    // i * (dqkv*total_seq) +j*total_seq  (total_seq, n_kv_h * dqkv)
    for i_seq_len in 0..seq_len{
            for i_n_groups in 0..n_groups {
                for i_tseq_len in 0..total_seq_len{  
                    for i_n_kv_h in 0..n_kv_h{
                        let mut sum:f32 = 0.;
                        for i_dqkv in 0..dqkv{
                            let _qval = _qdata[i_seq_len*(n_kv_h * n_groups * dqkv)+i_n_kv_h* (n_groups * dqkv)+i_n_groups*dqkv+i_dqkv];//[..total_seq_len]// i *
                            let _kval =  _kdata[i_tseq_len * (n_kv_h * dqkv)+i_n_kv_h*dqkv+i_dqkv];
                            sum += _qval*_kval;
                        }
                        // 单元赋值写法1
                        unsafe { // (n_kv_h, n_groups, seq, total_seq) 
                           // let _scoredata = ;
                            att_scores.data_mut()[i_n_kv_h*(n_groups*seq_len*total_seq_len)+i_n_groups*(seq_len*total_seq_len)+i_seq_len*total_seq_len+i_tseq_len] = sum/(dqkv as f32).sqrt();
                        }
                    }
                }
            }
    }

    // (n_kv_h, n_groups, seq, total_seq) 
    // attn = softmax(score)
    // 对于每个独立的“头”都得到一个 (seq_len, total_seq_len) 的权重矩阵
    for i_n_groups in 0..n_groups {
        for i_n_kv_h in 0..n_kv_h{      
            let mut _t = Tensor::new(att_scores.data()[i_n_kv_h*(n_groups*seq_len*total_seq_len)+i_n_groups*(seq_len*total_seq_len)..][..seq_len*total_seq_len].to_vec(), &vec![seq_len,total_seq_len]);
            OP::masked_softmax(&mut _t);     
            let dst = & mut unsafe{att_scores.data_mut()}
            [i_n_kv_h*(n_groups*seq_len*total_seq_len)+i_n_groups*(seq_len*total_seq_len)..]
            [..seq_len*total_seq_len];
            dst.copy_from_slice(_t.data());
        }
    }
    
    //attn_V = attn @ V   attn (n_kv_h, n_groups, seq, total_seq)  v:(total_seq, n_kv_h * dqkv)
         // -> (seq_len,n_heads * head_size -> n_kv_h * n_groups * dqkv)
         // qhead * len * dim
    
    // (seq, n_kv_h * n_groups * dqkv)
    for i_n_kv_h in 0..n_kv_h{
        for i_n_groups in 0..n_groups{
            for i_seq in 0..seq_len{
                for i_dqkv in 0..dqkv{
                    let mut sum = 0. as f32;
                    for i_t_seq in 0..total_seq_len{
                        let _attn = att_scores.data()
                        [i_n_kv_h*(n_groups * seq_len * total_seq_len)
                        + i_n_groups*(seq_len*total_seq_len)
                        + i_seq* total_seq_len
                        + i_t_seq
                        ];
                        // (total_seq, n_kv_h * dqkv)
                        let _v = v.data()
                        [i_t_seq*(n_kv_h * dqkv)
                        + i_n_kv_h*dqkv
                        + i_dqkv
                        ];
                        sum += _attn*_v;
                    }
                    // (seq, n_kv_h * n_groups * dqkv)
                    // 单元赋值写法2
                    let _d =  & mut unsafe{hidden_states.data_mut()}
                    [i_seq*(n_kv_h * n_groups * dqkv)
                    + i_n_kv_h*(n_groups * dqkv)+i_n_groups*dqkv+i_dqkv];
                    * _d = sum;
                }

            }
        }
    }
   
   

    }



fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // todo!("Implement mlp");
    // let mut tmp = vec![];
    // for i in 0..residual.shape()[residual.shape().len()-1]{
    //     tmp.push(1.);
    // }
    // let _tmp_w = Tensor::new(residual.data().to_vec(),residual.shape());
    // let r2:Tensor<f32> = Tensor::default(residual.shape()); &_tmp_w,
    OP::rms_norm(hidden_states,residual,rms_w,eps);
    OP::matmul_transb(gate,0.,hidden_states,w_gate,1.);
    OP::matmul_transb(up,0.,hidden_states,w_up,1.);
    // let act = up.clone();
    OP::swiglu(up,gate);
    // let output=residual.clone();

    OP::matmul_transb(residual,1.,up,w_down,1.);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    println!("{}",model.params.rms_att_w[0].data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}



#[test]
pub fn test_load() {
    use std::path::PathBuf;
    use std::io::{Write, BufWriter};
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    // let model = Llama::from_safetensors(model_dir);
    let model_file = std::fs::read(&model_dir.join("model.safetensors")).unwrap();
    let safetensor = SafeTensors::deserialize(&model_file).unwrap();

      let parameter_type = std::any::type_name::<SafeTensors>();
      println!("参数类型: {}", parameter_type);
  
      let output_path = model_dir.join("output_chat.txt");
      let file = File::create(&output_path).unwrap();
      let mut writer = BufWriter::new(file);
  
      writeln!(writer, "参数类型: {}", parameter_type).unwrap();
      writeln!(writer, "包含的张量数量: {}", safetensor.len()).unwrap();
  
      for (name, tensor) in safetensor.tensors() {
          writeln!(writer, "张量名称: {}", name).unwrap();
          writeln!(writer, "张量数据类型: {:?}", tensor.dtype()).unwrap();
          writeln!(writer, "张量shape : {:?}", tensor.shape()).unwrap();
      }
  
      println!("信息已成功写入到文件: {:?}", output_path);

}
