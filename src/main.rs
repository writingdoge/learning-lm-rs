mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod benchmark;

use std::path::PathBuf;
use std::sync::Arc;
use std::net::SocketAddr;
use axum::{
    routing::{post},
    Router,
    Json,
    extract::State,
};
use operators::FloatElement;
use params::FromF32Tensor;

use half::f16;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tower_http::cors::{CorsLayer, Any};

use dashmap::DashMap; // 用DashMap
use std::time::Instant;


use model::Llama;
use crate::kvcache::KVCache;

pub trait LlamaTrait<T: FloatElement>: Send + Sync {
    fn generate(
        &self,
        input_ids: &[u32],
        max_length: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kv_cache: &mut KVCache<T>,
    ) -> Vec<u32>;

    fn new_cache(&self) -> KVCache<T>;
}

impl LlamaTrait<f32> for Llama<f32> {
    fn generate(
        &self,
        input_ids: &[u32],
        max_length: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kv_cache: &mut kvcache::KVCache<f32>,
    ) -> Vec<u32> {
        self.generate(input_ids, max_length, top_p, top_k, temperature, kv_cache)
    }

    fn new_cache(&self) -> KVCache<f32> {
        self.new_cache()
    }
}

impl LlamaTrait<f16> for Llama<f16> {
    fn generate(
        &self,
        input_ids: &[u32],
        max_length: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        kv_cache: &mut KVCache<f16>,
    ) -> Vec<u32> {
        self.generate(input_ids, max_length, top_p, top_k, temperature, kv_cache)
    }

    fn new_cache(&self) -> KVCache<f16> {
        self.new_cache()
    }
}



struct AppState<T: FromF32Tensor + FloatElement> {
    llama: Arc<dyn LlamaTrait<T>>,
    tokenizer: Arc<Tokenizer>,
    session_kv_map: Arc<DashMap<String, KVCache<T>>>,
    session_turn_map: Arc<DashMap<String, u32>>,
    session_kvsnapshot_map: Arc<DashMap<String, Vec<KVCache<T>>>>,
}


#[derive(Deserialize)]
struct ChatRequest {
    pub session_id: String,
    pub message: String,
    pub max_len: Option<usize>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
}

#[derive(Deserialize)]
struct RollbackRequest {
    session_id: String,
    rollback_to_turn: Option<u32>, // 可选
}


#[derive(Serialize)]
struct RollbackResponse {
    success: bool,
    message: String, 
    current_turn: u32,
}

#[derive(Deserialize)]
struct DeleteSessionRequest {
    session_id: String,
}

#[derive(Serialize)]
struct DeleteSessionResponse {
    success: bool,
    message: String,
}

#[cfg(feature = "f32")]
fn init_state() -> Arc<AppState<f32>> {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models").join("chat");
    println!("Model directory: {:?}", model_dir);

    let model_load_start = Instant::now();
    let llama: Arc<dyn LlamaTrait<f32>> = Arc::new(Llama::<f32>::from_safetensors(&model_dir));
    let model_load_elapsed = model_load_start.elapsed();
    println!("模型加载完成, 用时: {:.2?}", model_load_elapsed);

    let tokenizer_load_start = Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    let tokenizer_load_elapsed = tokenizer_load_start.elapsed();
    println!("tokenizer 加载完成, 用时: {:.2?}", tokenizer_load_elapsed);


    Arc::new(AppState {
        llama,
        tokenizer,
        session_kv_map: Arc::new(DashMap::new()),
        session_turn_map: Arc::new(DashMap::new()),
        session_kvsnapshot_map: Arc::new(DashMap::new()),
    })
}

#[cfg(feature = "f16")]
fn init_state() -> Arc<AppState<half::f16>> {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models").join("chat");

    let model_load_start = Instant::now();
    let llama: Arc<dyn LlamaTrait<half::f16>> = Arc::new(Llama::<half::f16>::from_safetensors(&model_dir));
    let model_load_elapsed = model_load_start.elapsed();
    println!("模型加载完成, 用时: {:.2?}", model_load_elapsed);

    // let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let tokenizer_load_start = Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    let tokenizer_load_elapsed = tokenizer_load_start.elapsed();
    println!("tokenizer 加载完成, 用时: {:.2?}", tokenizer_load_elapsed);

    Arc::new(AppState {
        llama,
        tokenizer,
        session_kv_map: Arc::new(DashMap::new()),
        session_turn_map: Arc::new(DashMap::new()),
        session_kvsnapshot_map: Arc::new(DashMap::new()),
    })
}


async fn chat_handler<T: FloatElement + FromF32Tensor>(
    State(state): State<Arc<AppState<T>>>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let request_start = Instant::now();
    
    // 构建输入
    let input;
    let sys = "<|im_start|>system\nYou are a helpful assistant.";
    // You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.
    let user_begin = "<|im_end|>\n<|im_start|>user\n";
    let user_end = "<|im_end|>\n<|im_start|>assistant\n";

    let session_id = request.session_id.clone();
    let new_session = !state.session_kv_map.contains_key(&session_id);
    if new_session {
        println!("new session: {}", session_id);
        println!("creating a new KVCache for session: {}", session_id);
        input = format!("{}{}{}{}", sys, user_begin, request.message, user_end);
    } else {
        println!("session: {} has been created", session_id);
        println!("reusing existing KVCache for session: {}", session_id);
        input = format!("{}{}{}", user_begin, request.message, user_end);
    }

    let mut snapshots = state.session_kvsnapshot_map
    .entry(session_id.clone())
    .or_insert_with(|| Vec::new());

    
    let binding = state.tokenizer.encode(input.as_str(), true).unwrap();
    let input_ids = binding.get_ids();
    print!("会话 {}: {}",session_id, input.as_str());


    let mut kv = state.session_kv_map
        .entry(session_id.clone())
    .or_insert_with(|| state.llama.new_cache());

    let max_len = request.max_len.unwrap_or(250);
    let top_p = request.top_p.unwrap_or(0.8);
    let top_k = request.top_k.unwrap_or(30);
    let temperature = request.temperature.unwrap_or(1.0);


    let generation_start = Instant::now();
    

    let output_ids = state.llama.generate(
        input_ids,
        max_len,
        top_p,
        top_k,
        temperature,
        &mut kv,
    );

    let token_num = output_ids.len();

    let generation_elapsed = generation_start.elapsed();
    println!("模型生成耗时: {:.2?}", generation_elapsed);
    println!("sec per token: {:.2?}", generation_elapsed.as_secs_f64() / token_num as f64);
    
    let output = state.tokenizer.decode(&output_ids, true).unwrap();
    
    let request_elapsed = request_start.elapsed();
    println!("请求处理结束，会话: {}，总耗时: {:.2?}", session_id, request_elapsed);

        
    let turn_count = state.session_turn_map
    .entry(session_id.clone())
    .and_modify(|turn| { *turn += 1; })   
    .or_insert(1);                      
    println!("会话 {} 目前在 turn {}", session_id, *turn_count);

    if *turn_count % 1 == 0 { // 控制存储 snapshot 的间隔
        let snapshot = kv.create_snapshot();

        snapshots.push(snapshot);

        println!("创建快照 ：turn {}", *turn_count);
    }

    Json(ChatResponse {
        response: output,
    })
}

async fn rollback_handler<T: FloatElement + FromF32Tensor>(
    State(state): State<Arc<AppState<T>>>,
    Json(request): Json<RollbackRequest>,
) -> Json<RollbackResponse> {
    println!("ROLLBACK");
    let session_id = request.session_id.clone();
    let target_turn = request.rollback_to_turn.unwrap_or(1); 

    match rollback_session(&state, &session_id, target_turn) {
        Ok(current_turn) => {
            println!("Rollback successful for session {}", session_id);
            Json(RollbackResponse {
                success: true,
                message: format!("Rollback successful for session {}", session_id),
                current_turn
            })
        }
        Err((error,current_turn)) => {
            println!("Rollback failed: {}", error);
            Json(RollbackResponse {
                success: false,
                message: format!("Rollback failed: {}", error),
                current_turn
            })
        }
    }

}

fn rollback_session<T: FloatElement + FromF32Tensor>(
    state: &AppState<T>,
    session_id: &str,
    target_turn: u32,
) -> Result<u32, (String, u32)> {

    let current_turn = state.session_turn_map.get(session_id)
        .ok_or_else(|| ("Session turn not found".to_string(),0))?
        .clone(); 

    if target_turn <= 0 {
        return Err(("Failed: Illegal target turn".to_string(),current_turn));
    }

    if target_turn >= current_turn {
        println!("Test failed: Cannot rollback to future turn {} > {}", target_turn, current_turn);
        return Err(("Failed: Cannot rollback to future turn".to_string(),current_turn));
    }

    let snapshots = state.session_kvsnapshot_map.get(session_id)
        .ok_or_else(||( "No KV snapshots available".to_string(),current_turn))?;

    // if (snapshots.len() as u32) < target_turn{
    //     return Err("Failed: Cannot find target turn".to_string());
    // }

    let target_snapshot_index:usize = (target_turn - 1) as usize;
    // for i in 0..snapshots.len() {
    //     let snapshot_turn = (i as u32 + 1);
    //     if snapshot_turn <= target_turn {
    //         target_snapshot_index = i;
    //     }
    // }
    println!("Target snapshot index: {}", target_snapshot_index);
    let target_snapshot = &snapshots[target_snapshot_index];
    let target_len = target_snapshot.len();
    state.session_kv_map.alter(session_id, |_, mut kv_entry| { 
        kv_entry.load_snapshot(target_snapshot);
        kv_entry
    });
    drop(snapshots);

    let new_turn = target_snapshot_index as u32 + 1;
    state.session_turn_map.alter(session_id, |_, _| new_turn);

    state.session_kvsnapshot_map.alter(session_id, |_, mut snapshots| {
        snapshots.truncate(target_snapshot_index + 1);
        snapshots
    });

    assert_eq!(*state.session_turn_map.get(session_id).unwrap(), target_turn, "Turn should be 2 after rollback");
    assert_eq!(state.session_kvsnapshot_map.get(session_id).unwrap().len(), target_turn as usize, "Should have 2 snapshots after rollback");
    assert_eq!(state.session_kv_map.get(session_id).unwrap().value().len(), target_len, "KVCache length should be 25 after rollback to turn 2");

    println!("会话 {} 回滚到 turn {}", session_id, new_turn);
    Ok(new_turn)
}

async fn delete_session_handler<T: FloatElement + FromF32Tensor>(
    State(state): State<Arc<AppState<T>>>,
    Json(request): Json<DeleteSessionRequest>,
) -> Json<DeleteSessionResponse> {
    let session_id = request.session_id;
    
    // 删除所有与该会话相关的数据
    let kv_removed = state.session_kv_map.remove(&session_id).is_some();
    let turn_removed = state.session_turn_map.remove(&session_id).is_some();
    let snapshot_removed = state.session_kvsnapshot_map.remove(&session_id).is_some();
    
    if kv_removed || turn_removed || snapshot_removed {
        println!("会话 {} 已成功删除", session_id);
        Json(DeleteSessionResponse {
            success: true,
            message: format!("session {} is deleted", session_id),
        })
    } else {
        println!("会话 {} 不存在", session_id);
        Json(DeleteSessionResponse {
            success: false,
            message: format!("session {} doesn't exist", session_id),
        })
    }
}

#[tokio::main]
async fn main() {
    let state = init_state();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/chat", post(chat_handler))
        .route("/rollback", post(rollback_handler))
        .route("/delete_session", post(delete_session_handler))
        .layer(cors)
        .with_state(state);

    println!("服务器启动在 http://0.0.0.0:3000");
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    axum::serve(
        tokio::net::TcpListener::bind(addr)
            .await
            .unwrap(),
        app.into_make_service(),
    )
    .await
    .unwrap();
}

#[test]
fn test_init_state(){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    
    let model_load_start = Instant::now();
    let llama = Arc::new(model::Llama::<f32>::from_safetensors(&model_dir));
    let model_load_elapsed = model_load_start.elapsed();
    println!("模型加载完成, 用时: {:.2?}", model_load_elapsed);

    // let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let tokenizer_load_start = Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    let tokenizer_load_elapsed = tokenizer_load_start.elapsed();
    println!("tokenizer 加载完成, 用时: {:.2?}", tokenizer_load_elapsed);


}
