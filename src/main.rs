mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use eframe::egui;
use tokio::task;
use std::time::Instant;

use std::error::Error;
use std::fmt;

// 自定义错误类型
#[derive(Debug)]
struct LLMError(String);

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LLMError: {}", self.0)
    }
}

impl Error for LLMError {}


// **推理逻辑**
struct LLMInference {
    llama: Arc<Mutex<model::Llama<f32>>>,
    tokenizer: Arc<Tokenizer>,
}

impl LLMInference {
    fn new(model_path: &str) -> Self {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models").join(model_path);
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

        println!("Model {} loaded", model_path);
        println!("Tokenizer loaded");

        Self {
            llama: Arc::new(Mutex::new(llama)),
            tokenizer: Arc::new(tokenizer),
        }
    }

    fn run_inference(&self, input_text: &str) -> Result<String, LLMError> {
        let system_prompt = "<|im_start|>system\nYou are a highly knowledgeable and friendly assistant.<|im_end|>\n";
        let user_prefix = "<|im_start|>user\n";
        let assistant_prefix = "<|im_end|>\n<|im_start|>assistant\n";

        let input = format!("{}{}{}{}", system_prompt, user_prefix, input_text, assistant_prefix);

        // **1. 处理 tokenizer 可能的错误**
        let binding = self.tokenizer.encode(input.as_str(), true)
            .map_err(|e| LLMError(format!("Tokenizer 错误: {}", e)))?;

        let input_ids = binding.get_ids();

        let mut kv_cache = self.llama.lock().map_err(|_| LLMError("模型锁定失败".to_string()))?.new_cache();

        // **2. 处理模型推理可能的错误**
        let output_ids = self.llama.lock()
            .map_err(|_| LLMError("无法获取模型锁".to_string()))?
            .generate(input_ids, 256, 0.55, 35, 0.0, &mut kv_cache);

        let output_text = self.tokenizer.decode(&output_ids, true)
            .map_err(|e| LLMError(format!("解码失败: {}", e)))?;

        Ok(output_text)
    }
}

//     fn run_inference(&self, input_text: &str) -> String {
//         let system_prompt = "<|im_start|>system\nYou are a highly knowledgeable and friendly assistant.<|im_end|>\n";
//         let user_prefix = "<|im_start|>user\n";
//         let assistant_prefix = "<|im_end|>\n<|im_start|>assistant\n";

//         let input = format!("{}{}{}{}", system_prompt, user_prefix, input_text, assistant_prefix);
//         let binding = self.tokenizer.encode(input.as_str(), true).unwrap();
//         let input_ids = binding.get_ids();

//         let mut kv_cache = self.llama.lock().unwrap().new_cache();
//         let output_ids = self.llama.lock().unwrap().generate(input_ids, 256, 0.55, 35, 0.0, &mut kv_cache);
//         let output_text = self.tokenizer.decode(&output_ids, true).unwrap();

//         output_text
//     }
// }


// **对话应用结构**
struct LLMApp {
    llm: Arc<LLMInference>,
    user_input: String,
    messages: Arc<Mutex<Vec<(String, bool)>>>, // 存储聊天记录 (String, bool: true=用户, false=AI)
    last_runtime: Arc<Mutex<f32>>,             // 记录推理时长
}

impl LLMApp {
    fn new(llm: Arc<LLMInference>) -> Self {
        Self {
            llm,
            user_input: String::new(),
            messages: Arc::new(Mutex::new(Vec::new())),
            last_runtime: Arc::new(Mutex::new(0.0)),
        }
    }
}

impl eframe::App for LLMApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 AI Chat");

            ui.separator();

            // **对话窗口**
            egui::ScrollArea::vertical().show(ui, |ui| {
                let messages = self.messages.lock().unwrap();
                for (text, is_user) in messages.iter() {
                    if *is_user {
                        ui.horizontal(|ui| {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                                ui.add(egui::Label::new(format!("🧑‍💻 {}", text)).wrap(true));
                            });
                        });
                    } else {
                        // ui.horizontal(|ui| {
                        //     ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                        //         ui.add(egui::Label::new(format!("🤖 {}", text)).wrap(true));
                        //     });
                        // });
                        ui.horizontal(|ui| {
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                                // 限制此布局内的控件最大宽度为400像素
                                ui.set_max_width(400.0);
                                ui.add(egui::Label::new(format!("🤖 {}", text)).wrap(true));
                            });
                        });
                    }
                }
            });

            ui.separator();

            // **输入框**
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.user_input)
                    .hint_text("请输入内容..."));
                
                if ui.button("发送 🚀").clicked() {
                    let input = self.user_input.clone();
                    if input.is_empty() { return; }

                    let llm_clone = Arc::clone(&self.llm);
                    let messages = Arc::clone(&self.messages);
                    let last_runtime = Arc::clone(&self.last_runtime);

                    // **添加用户消息**
                    messages.lock().unwrap().push((input.clone(), true));
                    self.user_input.clear();

                    let start_time = Instant::now();
                    
                    // **启动异步推理**
                    // task::spawn_blocking(move || {
                    //     let response = llm_clone.run_inference(&input);
                    //     let duration = start_time.elapsed().as_millis() as f32;

                    //     match response {
                    //         Ok(result) => {
                    //             messages.lock().unwrap().push((result, false));
                    //             *last_runtime.lock().unwrap() = duration;
                    //         }
                    //         Err(e) => {
                    //             messages.lock().unwrap().push((format!("❌ 错误: {:?}", e), false));
                    //         }
                    //     }

                    //     ctx.request_repaint();
                    // });
                    task::spawn_blocking({
                        let messages = Arc::clone(&self.messages);
                        let last_runtime = Arc::clone(&self.last_runtime);
                        let llm_clone = Arc::clone(&self.llm);
                        let input = self.user_input.clone();
                    
                        move || {
                            let start_time = Instant::now();
                            let response = llm_clone.run_inference(&input);
                            let duration = start_time.elapsed().as_millis() as f32;
                    
                            match response {
                                Ok(result) => {
                                    messages.lock().unwrap().push((result, false));
                                    *last_runtime.lock().unwrap() = duration;
                                }
                                Err(e) => {
                                    messages.lock().unwrap().push((format!("❌ 错误: {:?}", e), false));
                                }
                            }
                    
                            // 不直接在子线程调用 ctx，而是依靠数据更新后下一个 update 自动刷新
                        }
                    });
                    
                    
                }
            });

            ui.label(format!("⏱️ 生成时间: {:.2} 毫秒", *self.last_runtime.lock().unwrap()));
        });
    }
}

// ✅ 运行 UI
fn main() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let llm = Arc::new(LLMInference::new("chat"));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    runtime.block_on(async {
        eframe::run_native(
            "AI Chat UI",
            options,
            Box::new(|_| Box::new(LLMApp::new(llm))),
        ).expect("Failed to start UI");
    });
}
