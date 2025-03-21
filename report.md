# 概要
功能完成情况如下：
- [x] 网络服务API 
- [ ] UI界面
- [x] 多会话管理和对话回滚 
- [x] 混合精度优化
- [x] CPU多线程优化+性能对比
- [ ] 适配NVDIA加速软件栈后端+GPU并行推理

# 基础功能

## 计算过程实现
使用 `Tensor<T>` 的 `data(),data_mut()`接口实现数据的访问与修改。

对于较复杂的部分，如 self-attention中 多组矩阵的存取、计算，也是多写几个循环，手动索引实现。没有做广播机制，没有用到`reshape()`。
## 故事续写与AI对话
故事续写效果如下：
![image.png](https://pic-1324265358.cos.ap-nanjing.myqcloud.com/20250321210407.png)

效果有些诡异，之前测试时和python transformers库对比过，得到调用相同模型的中间计算结果一致。
不过，存在重复输出问题，或许后续可考虑 repetition penalty 优化。

AI对话中，用`<|im_start|>`等提示词实现了 Jinja2 模板，和用户输入共同构建完整的输入。
效果如下：
![image.png](https://pic-1324265358.cos.ap-nanjing.myqcloud.com/20250321172112.png)

# 拓展功能

## 混合精度推理

系统支持在启动服务时，选择2种精度模式：
- **基础模式（全 fp32）**：全程使用 fp32，计算稳定，精度最高。
- **混合精度模式（fp16+fp32）**：
- 模型权重和 **KVCache** 以 fp16 存储
- 算子内部按需做 f16 -> f32 转换(`to_f32()`)
- 推理中间态基本以 fp32 存储。

**实现方式**：
本项目通过泛型和 Cargo feature 机制设计，结合 `half::f16` 实现了混合精度支持。在编译阶段即可确定使用 fp16 或 fp32，保证推理流程统一、代码无重复，且具备良好的扩展性与性能表现。

**实现效果**：
发现使用fp16的算子运算时间达到fp32的2倍以上，认为可能是由于转换为fp32的过程耗时。

效果如图：
![image.png](https://pic-1324265358.cos.ap-nanjing.myqcloud.com/20250321163929.png)

## 多线程优化

基于 Rust 的 rayon 库，对模型参数加载和推理计算过程进行了多线程并行优化，充分发挥 CPU 多核性能。

加速思路是，通过 `par_iter` 、 `par_chunks`2种方式，将数据按维度或 token 批次拆分成独立的计算单元，且注意避免 可变引用竞争。

例如在模型权重加载时，做了2层优化：
在读取tensor时，`par_chunks`，每4字节（1个float32数据）作为一个chunk读取。
对模型的每一层，并发调用 `get_tensor()` 加载对应的权重 tensor。
```rust
(0..config.num_hidden_layers).into_par_iter()
    .map(|i| get_tensor(...))
    .collect()
```

算子的优化思路如下：

| 算子                 | 优化方式                      | 说明                                    |
| ------------------ | ------------------------- | ------------------------------------- |
| **gather**         | `par_chunks_mut`          | 每个 dim 大小的块对应 indices中一个 token        |
| **RoPE**           | `par_chunks_mut`          | 每个 n_heads * d 大小的块对应一个token          |
| **RMSNorm**        | `par_chunks_mut`          | 按 token 并行                            |
| **masked softmax** | `par_chunks_mut`          | 按 seq_len * total_seq_len（对每个head） 并行 |
| **SwiGLU**         | `par_iter_mut + par_iter` | 逐元素并行即可                               |
| **matmul_transb**  | `par_chunks_mut`          | 每个输出矩阵行独立计算                           |



测试性能时，先和串行函数的结果比较正确性，在warmup后再进入正式测试阶段，计算测试的平均时间。

测试结果如下表：

| 算子                 | 原版平均时间 (ms) | 优化后平均时间 (ms) | 加速比 (原版 / 优化) |
| ------------------ | ----------- | ------------ | ------------- |
| **RMSNorm**        | 45.493      | 5.680        | **8.01x**     |
| **Gather**         | 20.421      | 2.188        | **9.34x**     |
| **Softmax**        | 272.774     | 194.225      | **1.40x**     |
| **SwiGLU**         | 0.880       | 0.587        | **1.50x**     |
| **MatMul_TransB**  | 9830.634    | 3775.616     | **2.60x**     |
| **Self-Attention** | 4.024       | 2.409        | **1.67x**     |

Softmax、SwiGLU、Self-Attention 优化效果有限，可能是受限于测试数据规模有限和内存访问速度。

## 多对话管理
支持创建对话，对话存储快照，回滚对话的操作。

选用的机制是，每个“会话”使用独立的 KVCache，在特定的 checkpoint 轮次（目前设置为每轮存一次）做KVCache备份，用哈希表存储。
回滚机制为：
1. 将此时的kvcache替换为某个已存储的 KVCache 备份，
2. 删除多余的快照信息。
在这里存在不同的实现机制，比如不删除之后的快照信息，实现保存走向不同的会话的功能，又或者存储历史对话文字信息（得到更好的记录和用户体验）。

实现细节是，使用了DashMap，一个支持并发的哈希表。另外，修改表内信息时，注意所有权的问题，防止死锁。

观察到随着 turn 数的上升，KVCache 参数量增大，推理速度下降。
**后续优化**：使用滑动窗口限制KVCache参数量。

## 网络服务 API

使用 Axum 框架实现了 RESTful 风格的网络服务，封装了模型推理、多会话管理与回滚功能，提供以下核心接口：
#### **1. /chat**
- **功能**：接收用户输入，接入已存在会话或开启新会话，调用模型生成回复。
- **请求参数**（JSON）：
```json
{   
"session_id": "abc123",   
"message": "Hello, how are you?",  
"max_len": 1024, 
"top_p": 0.8,   
"top_k": 30,   
"temperature": 1.0 
}
```
- 可选传入temperature等参数。
#### **2. /rollback**
- **功能**：将某个 `session_id` 的对话回滚至历史中的某一 `target_turn`，恢复到指定轮次的 KVCache 快照。
- **请求参数**（JSON）：
```json
{
  "session_id": "abc123",
  "target_turn": 2
}
```
- target_turn 默认为1
####  **3. /delete_session**
- **功能**：彻底删除某个会话及其对应的 KVCache 和历史快照，释放内存。
- **请求参数**（JSON）：
```json
{
  "session_id": "abc123"
}
```
使用`test.sh`测试。

![image.png](https://pic-1324265358.cos.ap-nanjing.myqcloud.com/20250321212909.png)

![image.png](https://pic-1324265358.cos.ap-nanjing.myqcloud.com/20250315170442.png)


**后续优化**：配合Gradio库搭UI界面。

# 总结
本次项目实现了：
- 核心大模型推理框架
- 混合精度支持
- 多线程并行优化
- 完整的多会话和对话回滚机制
- RESTful API 服务

