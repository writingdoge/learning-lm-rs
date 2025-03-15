// use crate::config;
// use crate::kvcache;
// use crate::model;
// use crate::operators;
// use crate::params;
// use crate::tensor;

// use std::path::PathBuf;
// use std::time::Duration;
// use std::sync::Arc;
// use axum::{
//     routing::{post, get},
//     Router, Json, extract::State,
//     http::StatusCode,
// };
// use serde::{Deserialize, Serialize};
// use tokio::sync::{Mutex, Semaphore};
// use tokenizers::Tokenizer;
// use tower_http::cors::CorsLayer;
// use tower::limit::RateLimitLayer;
// use tower::ServiceBuilder;

// #[derive(Deserialize)]
// pub struct RequestData {
//     input: String,
// }

// #[derive(Serialize)]
// pub struct ResponseData {
//     output: String,
// }

// #[derive(Serialize)]
// pub struct ErrorResponse {
//     error: String,
// }

// #[derive(Serialize)]
// pub struct HealthResponse {
//     status: String,
//     active_requests: usize,
// }

// struct AppState {
//     llama: Arc<model::Llama<f32>>,
//     tokenizer: Arc<Tokenizer>,
//     inference_semaphore: Arc<Semaphore>,
//     active_requests: Arc<Mutex<usize>>,
// }

// async fn health_check(
//     State(state): State<Arc<AppState>>,
// ) -> Json<HealthResponse> {
//     let active = *state.active_requests.lock().await;
//     Json(HealthResponse {
//         status: "OK".to_string(),
//         active_requests: active,
//     })
// }

// async fn generate_text(
//     State(state): State<Arc<AppState>>,
//     Json(payload): Json<RequestData>,
// ) -> Result<Json<ResponseData>, (StatusCode, Json<ErrorResponse>)> {
//     // 获取信号量许可
//     let _permit = state.inference_semaphore.acquire().await.map_err(|_| {
//         (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse {
//             error: "Server is too busy".to_string()
//         }))
//     })?;

//     // 增加活跃请求计数
//     {
//         let mut active = state.active_requests.lock().await;
//         *active += 1;
//     }

//     // 确保在函数结束时减少活跃请求计数
//     struct RequestGuard {
//         active_requests: Arc<Mutex<usize>>,
//     }
    
//     impl Drop for RequestGuard {
//         fn drop(&mut self) {
//             tokio::spawn({
//                 let active_requests = self.active_requests.clone();
//                 async move {
//                     let mut active = active_requests.lock().await;
//                     *active -= 1;
//                 }
//             });
//         }
//     }

//     let _guard = RequestGuard {
//         active_requests: state.active_requests.clone(),
//     };

//     // 在新线程中执行模型推理
//     let result = tokio::task::spawn_blocking({
//         let llama = state.llama.clone();
//         let tokenizer = state.tokenizer.clone();
//         let input = payload.input.clone();
        
//         move || {
//             let mut input_text = String::new();
//             let sys = "<|im_start|>system\nYou are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.\n";
//             let user_begin = "<|im_end|>\n<|im_start|>user\n"; 
//             let user_end = "<|im_end|>\n<|im_start|>assistant\n";
//             input_text = sys.to_string() + user_begin + input.as_str() + user_end;

//             let mut kv = llama.new_cache();
//             let binding = tokenizer.encode(input_text.as_str(), true).map_err(|e| {
//                 format!("Failed to encode input: {}", e)
//             })?;
//             let input_ids = binding.get_ids();

//             let output_ids = llama.generate(input_ids, 256, 0.55, 35, 0., &mut kv);
//             let output = tokenizer.decode(&output_ids, true).map_err(|e| {
//                 format!("Failed to decode output: {}", e)
//             })?;

//             Ok::<String, String>(output)
//         }
//     }).await.map_err(|e| {
//         (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
//             error: format!("Task execution error: {}", e)
//         }))
//     })??;

//     Ok(Json(ResponseData { output: result }))
// }

// pub async fn start_server() {
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("chat");

//     let llama = Arc::new(model::Llama::<f32>::from_safetensors(&model_dir));
//     println!("模型加载完成");
//     let tokenizer = Arc::new(Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap());
//     println!("tokenizer加载完成");

//     let state = Arc::new(AppState {
//         llama,
//         tokenizer,
//         // 限制最多4个并发推理请求
//         inference_semaphore: Arc::new(Semaphore::new(4)),
//         active_requests: Arc::new(Mutex::new(0)),
//     });

//     // let app = Router::new()
//     //     .route("/health", get(health_check))
//     //     .route("/generate", post(generate_text))
//     //     .layer(CorsLayer::permissive())
//     //     .layer(RateLimitLayer::new(10, Duration::from_secs(1)))
//     //     .with_state(state);

//     let app = Router::new()
//     .route("/health", get(health_check))
//     .route("/generate", post(generate_text))
//     .with_state(state) // 先注入状态
//     .layer(CorsLayer::permissive())
//     .layer(RateLimitLayer::new(10, Duration::from_secs(1)));


//     // use tower::ServiceBuilder;

//     // let middleware = ServiceBuilder::new()
//     // .layer(CorsLayer::permissive())
//     // .layer(RateLimitLayer::new(10, Duration::from_secs(1)));

//     // let app = Router::new()
//     // .route("/health", get(health_check))
//     // .route("/generate", post(generate_text))
//     // .with_state(state)
//     // .layer(middleware);


//     println!("API 服务器运行在 http://127.0.0.1:8000");
    
//     let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
//     axum::serve(listener, app).await.unwrap();
// }

use std::{path::PathBuf, sync::Arc, time::Duration};
use axum::{
    routing::{get, post},
    extract::State,
    http::StatusCode,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Semaphore};
use tokenizers::Tokenizer;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower::limit::RateLimitLayer;

// 请求 & 响应结构体
#[derive(Deserialize)]
pub struct RequestData {
    input: String,
}

#[derive(Serialize)]
pub struct ResponseData {
    output: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    active_requests: usize,
}

// 共享应用状态
struct AppState {
    inference_semaphore: Arc<Semaphore>,
    active_requests: Arc<Mutex<usize>>,
}

// 健康检查处理函数
async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let active = *state.active_requests.lock().await;
    Json(HealthResponse {
        status: "OK".to_string(),
        active_requests: active,
    })
}

// 生成文本的请求处理函数
async fn generate_text(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RequestData>,
) -> Result<Json<ResponseData>, (StatusCode, Json<ErrorResponse>)> {
    let _permit = state.inference_semaphore.acquire().await.map_err(|_| {
        (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse {
            error: "Server is too busy".to_string(),
        }))
    })?;

    // 计数管理
    let mut active = state.active_requests.lock().await;
    *active += 1;
    drop(active); // 释放锁，减少锁定时间

    // 这里省略了实际的模型调用，仅返回示例文本
    let result = format!("Processed input: {}", payload.input);

    // 任务结束后减少计数
    let mut active = state.active_requests.lock().await;
    *active -= 1;

    Ok(Json(ResponseData { output: result }))
}

// 服务器启动函数
pub async fn start_server() {
    let state = Arc::new(AppState {
        inference_semaphore: Arc::new(Semaphore::new(4)), // 限制最多4个并发请求
        active_requests: Arc::new(Mutex::new(0)),
    });

    // 1️⃣ **定义路由**
    let router = Router::new()
        .route("/health", get(health_check))
        .route("/generate", post(generate_text))
        .with_state(state);

    // 2️⃣ **将 `RateLimitLayer` 作用于 `ServiceBuilder`，避免影响 Router**
    let rate_limited_router = ServiceBuilder::new()
        .layer(RateLimitLayer::new(10, Duration::from_secs(1))) // 每秒最多10个请求
        .service(router);

    // 3️⃣ **在外层作用 `CorsLayer`，并合并**
    let app = Router::new()
        .merge(rate_limited_router)
        .layer(CorsLayer::permissive()); // 允许跨域

    println!("API 服务器运行在 http://127.0.0.1:8000");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
