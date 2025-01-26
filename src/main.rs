mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

// #[test]
fn main() {
    // 加载模型
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    println!("模型加载完成");
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    println!("tokenizer加载完成");
    // // 输入Input：

    let mut input = String::new();
    let b1 = std::io::stdin().read_line(&mut input).unwrap();

    input = input.trim().to_string();
    println!("input: {}  size:{}", input,b1);
    // // 建造模板
    let sys = "<|im_start|>system\nYou are a highly knowledgeable and friendly assistant. Your goal is to understand and respond to user inquiries with clarity. Your interactions are always respectful, helpful, and focused on delivering the most accurate information to the user.";
    let user_begin = "<|im_end|>\n<|im_start|>user\n";
    // 
    let user_end = "<|im_end|>\n<|im_start|>assistant\n";

    input = sys.to_string()+ user_begin + input.as_str();

    input = input + user_end;

    println!("input: {}", input);

    let mut kv=llama.new_cache();

    loop{
   

    println!("input: {}  size:{}", input,b1);
    let binding = tokenizer.encode(input.as_str(), true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input.as_str());
    // 传入一句话
    let output_ids = llama.generate(
        input_ids,
        256,
        0.55,
        35,
        0.,
        & mut kv,
    );

    let output = tokenizer.decode(&output_ids, true).unwrap();
    println!("{}", output);

    println!("请输入:");

    input.clear();

    let b1 = std::io::stdin().read_line(&mut input).unwrap();
    input = input.trim().to_string();
    if input.trim() == "exit"{
        break;
    }

    input =  user_begin.to_string() + input.as_str() + user_end;


}
    
    // println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

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