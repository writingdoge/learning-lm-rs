mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use std::sync::Arc;
use std::net::SocketAddr;
use axum::{
    routing::{post},
    Router,
    Json,
    extract::State,
};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tower_http::cors::{CorsLayer, Any};

use dashmap::DashMap; // 用DashMap
use std::time::Instant;

struct AppState {
    llama: Arc<model::Llama<f32>>,
    tokenizer: Arc<Tokenizer>,
    // kv_cache: Arc<std::sync::Mutex<kvcache::KVCache<f32>>>,
    session_kv_map: Arc<DashMap<String, kvcache::KVCache<f32>>>,
    session_turn_map: Arc<DashMap<String, u32>>,
    session_kvsnapshot_map: Arc<DashMap<String, Vec<kvcache::KVCache<f32>>>>,
}


#[derive(Deserialize)]
struct ChatRequest {
    session_id: String, 
    message: String, 
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

// 回滚响应结构
#[derive(Serialize)]
struct RollbackResponse {
    success: bool,
    message: String, 
    current_turn: u32,
}

fn init_state()->Arc<AppState>{
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    // let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let model_load_start = Instant::now();
    let llama = Arc::new(model::Llama::<f32>::from_safetensors(&model_dir));
    let model_load_elapsed = model_load_start.elapsed();
    println!("模型加载完成, 用时: {:.2?}", model_load_elapsed);

    // let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let tokenizer_load_start = Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    let tokenizer_load_elapsed = tokenizer_load_start.elapsed();
    println!("tokenizer 加载完成, 用时: {:.2?}", tokenizer_load_elapsed);

    let session_kv_map = Arc::new(DashMap::new());
    let session_turn_map = Arc::new(DashMap::new());
    let session_kvsnapshot_map = Arc::new(DashMap::new());


    Arc::new(AppState {
        llama,
        tokenizer,
        session_kv_map,
        session_turn_map,
        session_kvsnapshot_map,
    })

}

#[tokio::main]
async fn main() {

    let state = init_state();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/chat", post(chat_handler)) //对话
        .route("/rollback", post(rollback_handler)) //回滚
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
   // roll_back();
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
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
    print!("session {}: {}",session_id, input.as_str());


    let mut kv = state.session_kv_map
    .entry(session_id.clone())
    .or_insert_with(|| state.llama.new_cache());


    let generation_start = Instant::now();
    
    // 硬编码输入参数，需修改
    let output_ids = state.llama.generate(
        input_ids,
        250,
        0.8,
        30,
        1.,
        &mut kv,
    );

    let generation_elapsed = generation_start.elapsed();
    println!("模型生成耗时: {:.2?}", generation_elapsed);
    
    let output = state.tokenizer.decode(&output_ids, true).unwrap();
    
    let request_elapsed = request_start.elapsed();
    println!("请求处理结束，会话: {}，总耗时: {:.2?}", session_id, request_elapsed);

        
    let turn_count = state.session_turn_map
    .entry(session_id.clone())
    .and_modify(|turn| { *turn += 1; })   
    .or_insert(1);                      
    println!("Session {} is now at turn {}", session_id, *turn_count);

    if *turn_count % 1 == 0 { // 控制存储 snapshot 的间隔
        let snapshot = kv.create_snapshot();

        snapshots.push(snapshot);

        println!("Checkpoint created at turn: {}", *turn_count);
    }

    Json(ChatResponse {
        response: output,
    })
}

// 回滚处理函数
async fn rollback_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RollbackRequest>,
) -> Json<RollbackResponse> {
    println!("ROLLBACK");
    let session_id = request.session_id.clone();
    let target_turn = request.rollback_to_turn.unwrap_or(1); // 如果未提供，默认为 1

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

fn rollback_session(state: &AppState, session_id: &str, target_turn: u32) -> Result<u32, (String,u32)> {

    println!("begin getting current_turn");
    let current_turn = state.session_turn_map.get(session_id)
        .ok_or_else(|| ("Session turn not found".to_string(),0))?
        .clone(); 
    println!("OK getting current_turn");

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

    println!("Rollback successful: session {} rollback to turn {}", session_id, new_turn);
    Ok((new_turn))
}


// 续写部分
// fn main() {
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
//     let input = "Once upon a time";
//     let binding = tokenizer.encode(input, true).unwrap();
//     let input_ids = binding.get_ids();
//     print!("\n{}", input);
//     let output_ids = llama.generate(
//         input_ids,
//         500,
//         0.4,
//         30,
//         0.,
//     );
//     println!("{}", tokenizer.decode(&output_ids, true).unwrap());
// }

// #[test]
// pub fn test_chat() {
//     chat();
// }


#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use axum::Json;
    use std::sync::Arc;
    use tokio;
    use std::time::Instant;

    #[tokio::test] // 异步测试
    async fn roll_back() {
        let state = init_state();

        let session_id = "test-rollback-session".to_string();
        let mut kv = state.llama.new_cache();
        let mut snapshots = Vec::new();

        println!("Creating snapshots for testing...");
        state.session_turn_map.insert(session_id.clone(), 0);

        println!("Simulating turn 1...");
        kv.increment(10);
        state.session_turn_map.alter(&session_id, |_, _| 1);
        snapshots.push(kv.create_snapshot());

        println!("Simulating turn 2...");
        kv.increment(15);
        state.session_turn_map.alter(&session_id, |_, turn| turn + 1);
        snapshots.push(kv.create_snapshot());

        println!("Simulating turn 3...");
        kv.increment(20);
        state.session_turn_map.alter(&session_id, |_, turn| turn + 1);
        snapshots.push(kv.create_snapshot());

        state.session_kv_map.insert(session_id.clone(), kv);
        state.session_kvsnapshot_map.insert(session_id.clone(), snapshots);

        println!("Setup complete. Starting rollback tests...");

        println!("\nTest 1: Rollback to turn 2");
        let rollback_request1 = RollbackRequest {
            session_id: session_id.clone(),
            rollback_to_turn: Some(2),
        };
        let response1 = rollback_handler(State(state.clone()), Json(rollback_request1)).await;
        let response_body1 = response1.0;

        assert!(response_body1.success, "Rollback to turn 2 should succeed");
        assert_eq!(response_body1.current_turn, 2, "Current turn should be 2");

        println!("\nTest 2: Rollback to non-existent turn (should fail)");
        let rollback_request2 = RollbackRequest {
            session_id: session_id.clone(),
            rollback_to_turn: Some(10),
        };
        let response2 = rollback_handler(State(state.clone()), Json(rollback_request2)).await;
        let response_body2 = response2.0;

        assert!(!response_body2.success, "Rollback to turn 10 should fail");
        assert_eq!(
            response_body2.message,
            "Rollback failed: Failed: Cannot rollback to future turn",
            "Error message should indicate failure"
        );
        assert_eq!(
            response_body2.current_turn, 2,
            "Even if rollback fails, current_turn should remain unchanged"
        );

        println!("\nTest 3: Default rollback (to previous snapshot)");
        let rollback_request3 = RollbackRequest {
            session_id: session_id.clone(),
            rollback_to_turn: None, // 默认回滚
        };
        let response3 = rollback_handler(State(state.clone()), Json(rollback_request3)).await;
        let response_body3 = response3.0;

        assert!(response_body3.success, "Default rollback should succeed");
        assert_eq!(response_body3.current_turn, 1, "Current turn should rollback to previous snapshot");

        println!("rollback tests OK");
    }
}
