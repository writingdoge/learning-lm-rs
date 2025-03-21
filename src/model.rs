use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::{LLamaParams, FromF32Tensor};
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;

use crate::operators::FloatElement;

use rayon::prelude::*;

use std::io::Write;

pub struct Llama<T: FromF32Tensor + FloatElement> {
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

pub fn write_config_to_file<P: AsRef<Path>>(config: &LlamaConfigJson, path: P) -> std::io::Result<()> {
    // 序列化为 JSON 字符串，格式化输出
    let json_string = serde_json::to_string_pretty(config)
        .expect("Failed to serialize LlamaConfigJson");

    // 创建并写入文件
    let mut file = File::create(path)?;
    file.write_all(json_string.as_bytes())?;

    Ok(())
}

impl <T: FloatElement+ FromF32Tensor> Llama<T> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let mut config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
       // config.use_fp16 = Some(true);

        write_config_to_file(&config, "llama_config.json").unwrap();

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

    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h; // 

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
       

        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup

        // 看起来内部可以用f16，但是因为residual是f32的，还是要转为f32
        OP::gather(&mut residual, input, &self.params.embedding_table);

        // println!("residual {:?}",residual.data()[0]);
        // println!("{}",self.n_layers);
        for layer in 0..self.n_layers {
            // println!("{} ",layer);
            // println!("residual {:?}",residual.data()[0]);

            OP::rms_norm_parallel(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            // cb用f16 内部也f16计算
            OP::matmul_transb_parallel(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb_parallel(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb_parallel(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            // 内部转f32计算，再存回f16
            OP::rope_parallel(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope_parallel(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            // f32 
            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // println!("q {:?}",q.data()[0])
            // 输入的q,k,v为f16,转为f32计算，输出f32
            self_attention_parallel(& mut hidden_states,& mut att_scores,
            q,full_k,full_v,
            self.n_kv_h,n_groups,seq_len,total_seq_len,self.dqkv);


           // todo!("down_proj matmul and add residual");

             // (seq, n_kv_h * n_groups * dqkv)
            // out = attn_V @ O_weight.T     (q_head,len,dim)   wo:(hidden_size, n_heads * head_size)
    
            // C = beta * C + alpha * A @ B^T
            // residual = out + residual
        
            // f16 
            let wo = & self.params.wo[layer];

            //内部全是f32
            OP::matmul_transb_parallel(&mut residual, 1., & hidden_states,wo , 1.);

            // println!("{:?}",residual.shape());
            // println!("after self-attn : {:?}",residual.data()[0]);
           // 内部f16转为f32计算,输出f32
           mlp(&mut residual,&mut hidden_states,
            &mut gate_buf,&mut up_buf,&self.params.w_up[layer],
        &self.params.w_down[layer],
    &self.params.w_gate[layer],&self.params.rms_ffn_w[layer],self.eps);
        }

        // println!("after mlp : {:?}",residual.data()[0]);

        // println!("yeah everything else is OK");
        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.

        // f32
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1,self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![1,self.d]);
        // seq_len, self.d

        // println!("last norm: input {:?}",residual.data()[0]);
        OP::rms_norm_parallel(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );
        // println!("output {:?}",hidden_states.data()[0]);

        // 内部用f32计算
        OP::matmul_transb_parallel(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits // 概率分布
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kv: &mut KVCache<T>,
    ) -> Vec<u32>{
        let mut result = Vec::<u32>::new();

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

pub fn self_attention_parallel<T: FloatElement>(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq) 
    q: &Tensor<T>,                   // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<T>,                   // (total_seq, n_kv_h * dqkv)
    v: &Tensor<T>,                   // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let _qdata = q.data();
    let _kdata = k.data();

    unsafe {
    att_scores
        .data_mut()
        .par_chunks_mut(n_groups * seq_len * total_seq_len)
        .enumerate()
        .for_each(|(i_n_kv_h, score_chunk)| {
            for i_n_groups in 0..n_groups {
                for i_seq in 0..seq_len {
                    for i_tseq in 0..total_seq_len {
                        let mut sum = 0.0;
                        for i_dqkv in 0..dqkv {
                            let q_idx = i_seq * (n_kv_h * n_groups * dqkv)
                                + i_n_kv_h * (n_groups * dqkv)
                                + i_n_groups * dqkv
                                + i_dqkv;
                            let k_idx = i_tseq * (n_kv_h * dqkv) + i_n_kv_h * dqkv + i_dqkv;
                            sum += _qdata[q_idx].to_f32() * _kdata[k_idx].to_f32();
                        }
                        let score_idx = i_n_groups * seq_len * total_seq_len + i_seq * total_seq_len + i_tseq;
                        score_chunk[score_idx] = sum / (dqkv as f32).sqrt();
                    }
                }
            }
        });

    // 按(n_kv_h*n_groups)级别并行，每个 block 独立做 softmax
    att_scores
        .data_mut()
        .par_chunks_mut(seq_len * total_seq_len)
        .for_each(|chunk| {
            let mut chunk_vec = Tensor::new(chunk.to_vec(),&vec![seq_len,total_seq_len]);
            OP::masked_softmax_parallel(&mut chunk_vec);
            chunk.copy_from_slice(chunk_vec.data()); 
        });
   

    let _vdata = v.data();
    hidden_states
        .data_mut()
        .par_chunks_mut(n_kv_h * n_groups * dqkv)
        .enumerate()
        .for_each(|(i_seq, out_chunk)| {
            for i_n_kv_h in 0..n_kv_h {
                for i_n_groups in 0..n_groups {
                    for i_dqkv in 0..dqkv {
                        let mut sum = 0.0;
                        for i_tseq in 0..total_seq_len {
                            let att_idx = i_n_kv_h * (n_groups * seq_len * total_seq_len)
                                + i_n_groups * (seq_len * total_seq_len)
                                + i_seq * total_seq_len
                                + i_tseq;
                            let v_idx = i_tseq * (n_kv_h * dqkv) + i_n_kv_h * dqkv + i_dqkv;
                            sum += att_scores.data()[att_idx] * _vdata[v_idx].to_f32();
                        }
                        let out_idx = i_n_kv_h * (n_groups * dqkv) + i_n_groups * dqkv + i_dqkv;
                        out_chunk[out_idx] = sum;
                    }
                }
            }
        });
    }
}

fn self_attention<T: FloatElement>(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq) 
    q: &Tensor<T>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
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
    // 对于每个独立的"头"都得到一个 (seq_len, total_seq_len) 的权重矩阵
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
                            // let _q_val32 = _qval.to_f32();

                            sum += _qval.to_f32()*_kval.to_f32();
                        }
                        // 赋值写法1
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
    // 对于每个独立的"头"都得到一个 (seq_len, total_seq_len) 的权重矩阵
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
                        sum += _attn*_v.to_f32();
                    }
                    // (seq, n_kv_h * n_groups * dqkv)
                    // 赋值写法2
                    let _d =  & mut unsafe{hidden_states.data_mut()}
                    [i_seq*(n_kv_h * n_groups * dqkv)
                    + i_n_kv_h*(n_groups * dqkv)+i_n_groups*dqkv+i_dqkv];
                    * _d = sum;
                }

            }
        }
    }
   
   

}



fn mlp<T: FloatElement>(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: f32,
) {
    // todo!("Implement mlp");
    OP::rms_norm(hidden_states,residual,rms_w,eps);
    // b,c是f16
    OP::matmul_transb(gate,0.,hidden_states,w_gate,1.);
    OP::matmul_transb(up,0.,hidden_states,w_up,1.);

    // 全是f16
    OP::swiglu(up,gate);

    // b是f16
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

use rand::{SeedableRng, rngs::StdRng, Rng};
use crate::benchmark::{random_tensor_f32,benchmark,random_tensor_f16};

#[test]
pub fn test_self_attention_opt() {
    let seq_len = 8;
    let total_seq_len = 8;
    let n_kv_h = 4;
    let n_groups = 2;
    let dqkv = 64;
    
    let q_shape = vec![seq_len, n_kv_h * n_groups * dqkv];
    let k_shape = vec![total_seq_len, n_kv_h * dqkv];
    let v_shape = k_shape.clone();
    let score_shape = vec![n_kv_h, n_groups, seq_len, total_seq_len];
    let out_shape = vec![seq_len, n_kv_h * n_groups * dqkv];

    let seed = 42;
    
    let q = random_tensor_f32(&q_shape, seed);
    let k = random_tensor_f32(&k_shape, seed + 1);
    let v = random_tensor_f32(&v_shape, seed + 2);

    let mut hidden_origin = Tensor::<f32>::default(&out_shape);
    let mut hidden_parallel = Tensor::<f32>::default(&out_shape);
    let mut att_scores_origin = Tensor::<f32>::default(&score_shape);
    let mut att_scores_parallel = Tensor::<f32>::default(&score_shape);

    self_attention(
        &mut hidden_origin,
        &mut att_scores_origin,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    self_attention_parallel(
        &mut hidden_parallel,
        &mut att_scores_parallel,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    let diff = hidden_origin
        .data()
        .iter()
        .zip(hidden_parallel.data().iter())
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0_f64, |acc, x| f64::max(acc, x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-4);

    let origin = benchmark(|| {
        self_attention(
            &mut hidden_origin,
            &mut att_scores_origin,
            &q,
            &k,
            &v,
            n_kv_h,
            n_groups,
            seq_len,
            total_seq_len,
            dqkv,
        );
    }, 5, 20);

    let opt = benchmark(|| {
        self_attention_parallel(
            &mut hidden_parallel,
            &mut att_scores_parallel,
            &q,
            &k,
            &v,
            n_kv_h,
            n_groups,
            seq_len,
            total_seq_len,
            dqkv,
        );
    }, 5, 20);

    println!("原版平均时间: {:.3}ms | 优化后平均时间: {:.3}ms", origin, opt);

}
// 原版平均时间: 4.024ms | 优化后平均时间: 2.409ms
#[test]
pub fn test_self_attention_opt_fp16() {
    let seq_len = 8;
    let total_seq_len = 8;
    let n_kv_h = 4;
    let n_groups = 2;
    let dqkv = 64;
    
    let q_shape = vec![seq_len, n_kv_h * n_groups * dqkv];
    let k_shape = vec![total_seq_len, n_kv_h * dqkv];
    let v_shape = k_shape.clone();
    let score_shape = vec![n_kv_h, n_groups, seq_len, total_seq_len];
    let out_shape = vec![seq_len, n_kv_h * n_groups * dqkv];

    let seed = 42;
    
    // 使用f16版本的输入
    let q = random_tensor_f16(&q_shape, seed);
    let k = random_tensor_f16(&k_shape, seed + 1);
    let v = random_tensor_f16(&v_shape, seed + 2);

    let mut hidden_origin = Tensor::<f32>::default(&out_shape);
    let mut hidden_parallel = Tensor::<f32>::default(&out_shape);
    let mut att_scores_origin = Tensor::<f32>::default(&score_shape);
    let mut att_scores_parallel = Tensor::<f32>::default(&score_shape);

    self_attention(
        &mut hidden_origin,
        &mut att_scores_origin,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    self_attention_parallel(
        &mut hidden_parallel,
        &mut att_scores_parallel,
        &q,
        &k,
        &v,
        n_kv_h,
        n_groups,
        seq_len,
        total_seq_len,
        dqkv,
    );

    let diff = hidden_origin
        .data()
        .iter()
        .zip(hidden_parallel.data().iter())
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0_f64, |acc, x| f64::max(acc, x));

    println!("最大误差 {:.6}", diff);
    assert!(diff < 1e-2);  

    let origin = benchmark(|| {
        self_attention(
            &mut hidden_origin,
            &mut att_scores_origin,
            &q,
            &k,
            &v,
            n_kv_h,
            n_groups,
            seq_len,
            total_seq_len,
            dqkv,
        );
    }, 5, 20);

    let opt = benchmark(|| {
        self_attention_parallel(
            &mut hidden_parallel,
            &mut att_scores_parallel,
            &q,
            &k,
            &v,
            n_kv_h,
            n_groups,
            seq_len,
            total_seq_len,
            dqkv,
        );
    }, 5, 20);

    println!("FP16原版平均时间: {:.3}ms | FP16优化后平均时间: {:.3}ms", origin, opt);
}
// FP16原版平均时间: 14.232ms | FP16优化后平均时间: 5.425ms