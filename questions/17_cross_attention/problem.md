# Cross Attention

## 背景

交叉注意力让一个序列从另一个序列读取信息，常用于编码器—解码器模型和多模态模型。查询长度与上下文长度可以不同。 本题要求只使用 NumPy 完成核心计算，以便直接观察张量形状和数值过程。

## 需求

实现 `cross_attention`。Q 只由 query 投影，K/V 只由 context 投影。按 num_heads 分头计算缩放点积注意力，再拼接并输出投影。mask 的第一维对应查询位置，第二维对应上下文位置，True 表示允许读取。 所有输入均保证合法，不需要处理题目约定以外的类型、形状或取值。

## 输入

- `query`：形状为 (query_length, model_dim)。
- `context`：形状为 (context_length, model_dim)。
- `w_q / w_k / w_v / w_o`：形状均为 (model_dim, model_dim)。
- `num_heads`：能整除模型维度的头数。
- `mask`：None 或形状为 (Q,K) 的布尔掩码。

除特别说明外，参与数值计算的数组和常量统一按 `float64` 处理。函数不应修改调用者传入的数组。

## 输出

返回形状为 `(query_length, model_dim)` 的 float64 数组。

## 数学定义

$A=\operatorname{softmax}(Q_{query}K_{context}^T/\sqrt{d_h})$，输出为 $AV_{context}$。

## 测试约定

固定随机种子生成 100 组输入输出，覆盖不等长 Q/KV、不同头数、随机稀疏掩码、无掩码、单查询位置。本题判定误差为 `rtol=1e-8`、`atol=1e-10`。测试只调用公开函数，不依赖标准输出；提交代码中可以定义私有辅助函数。
