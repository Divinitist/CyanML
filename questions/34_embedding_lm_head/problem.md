# Embedding + LM Head

## 背景

语言模型先把离散 token 下标查表为向量，经过隐藏变换后再由 LM Head 投影回词表。输入嵌入与输出权重共享可以减少参数，并让两个空间直接对齐。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `embedding_lm_head`。用 token_ids 对 embedding_weight 做 NumPy 高级索引得到 embeddings；hidden_states=tanh(embeddings@hidden_projection)。tie_weights=True 时忽略 lm_head_weight，使用 embedding_weight 作为输出矩阵；否则使用 lm_head_weight。LM Head 按 hidden_states@output_weight.T 计算，并在 bias 非 None 时相加。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `token_ids`：任意前导形状的整数 token 数组，元素位于词表范围内。
- `embedding_weight`：形状为 (vocab_size,model_dim) 的查表矩阵。
- `hidden_projection`：形状为 (model_dim,model_dim) 的隐藏变换。
- `lm_head_weight`：未共享时使用的 (vocab_size,model_dim) 权重。
- `lm_head_bias`：None 或形状为 (vocab_size,) 的偏置。
- `tie_weights`：是否共享输入嵌入权重。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(embeddings, hidden_states, logits)`，前两者末维为 model_dim，logits 末维为 vocab_size。

## 数学定义

$E=W_{emb}[ids]$，$H=\tanh(EW_h)$，$logits=HW_{out}^T+b$；共享时 $W_{out}=W_{emb}$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖一维/二维 token、重复 token、共享/独立权重、有/无偏置、不同词表和模型维度。本题误差标准为 `rtol=1e-10`、`atol=1e-12`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
