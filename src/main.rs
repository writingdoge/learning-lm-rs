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

// 定义应用状态
struct AppState {
    llama: Arc<model::Llama<f32>>,
    tokenizer: Arc<Tokenizer>,
    kv_cache: Arc<std::sync::Mutex<kvcache::KVCache<f32>>>,
}

// 请求体结构
#[derive(Deserialize)]
struct ChatRequest {
    message: String,
}

// 响应体结构
#[derive(Serialize)]
struct ChatResponse {
    response: String,
}

#[tokio::main]
async fn main() {
    // 加载模型
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    // let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let llama = Arc::new(model::Llama::<f32>::from_safetensors(&model_dir));
    println!("模型加载完成");
    // let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
    println!("tokenizer加载完成");


    // let mut input = String::new();
    // let b1 = std::io::stdin().read_line(&mut input).unwrap();
    // 把输入部分，移到handler里，多个混用一个kvcache
    // let mut kv=llama.new_cache();
    let kv_cache = Arc::new(std::sync::Mutex::new(llama.new_cache()));

    // 创建应用state
    let state = AppState {
        llama,
        tokenizer,
        kv_cache,
    };

    // 配置CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);


    // 创建路由
    let app = Router::new()
        .route("/chat", post(chat_handler))
        .layer(cors)
        .with_state(Arc::new(state));

    // 启动服务器
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

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Json<ChatResponse> {
    // 构建输入
    let sys = "<|im_start|>system\n";
    // You are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.
    let user_begin = "<|im_end|>\n<|im_start|>user\n";
    let user_end = "<|im_end|>\n<|im_start|>assistant\n";
    
    let input = format!("{}{}{}{}", sys, user_begin, request.message, user_end);
    
    let binding = state.tokenizer.encode(input.as_str(), true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input.as_str());
    
    let mut kv = state.kv_cache.lock().unwrap();
    
    // 硬编码输入参数，需修改
    let output_ids = state.llama.generate(
        input_ids,
        256,
        0.55,
        35,
        0.,
        &mut kv,
    );
    
    let output = state.tokenizer.decode(&output_ids, true).unwrap();
    
    Json(ChatResponse {
        response: output,
    })
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