# 自回归生成

## 背景

自回归生成反复取最后位置 logits 选择下一个 token。缓存模式只计算新 token 的 Q/K/V，非缓存模式每步重算整个前缀；在没有随机采样时，两条路径应产生相同结果。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `autoregressive_generate`。实现一个单层因果自注意力语言模型的贪心生成。params 包含 w_q/w_k/w_v/w_o，注意力输出与 embedding 残差相加，再与 embedding_weight 共享 LM Head。每步选择 argmax（并列取较小下标），生成 end_token 时立即停止。use_cache=True 时预填充 prompt 并追加 KV；False 时每步重算完整序列。返回每个决策步骤使用的 logits。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `prompt_tokens`：形状为 (prompt_length,) 的非空整数前缀。
- `embedding_weight`：形状为 (vocab_size,model_dim) 的共享嵌入/LM Head。
- `params`：含四个 (model_dim,model_dim) 注意力矩阵的字典。
- `num_heads`：能整除 model_dim 的查询/KV 头数。
- `max_new_tokens`：最多生成的 token 数。
- `end_token`：生成后提前停止的合法 token 下标。
- `use_cache`：是否启用 KV Cache 增量路径。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(tokens, step_logits)`；tokens 是包含原 prompt 和新 token 的列表，step_logits 形状为 `(实际生成步数,vocab_size)`。

## 数学定义

每步 $t_{next}=argmax(logits_{last})$。缓存路径保存 $K_{1:t},V_{1:t}$，新查询只与已缓存键值计算；它应与完整因果前向的最后位置相等。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖缓存/非缓存、单/多 token prompt、不同生成上限、提前 EOS、多个头数、不同词表以及相同参数下两路径一致性。本题误差标准为 `rtol=1e-8`、`atol=1e-10`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
