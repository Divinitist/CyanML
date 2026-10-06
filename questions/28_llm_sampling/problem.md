# LLM 采样

## 背景

语言模型把下一 token 表示为词表上的 logits。温度、Top-K 和 Top-P 依次改变候选分布，在确定性与多样性之间进行控制；固定随机种子使采样可复现。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `sample_next_token`。先将 logits 除以 temperature，再保留最大的 top_k 项；top_k=0 表示不过滤。随后在当前候选上按概率降序保留累计概率首次达到 top_p 所需的最小集合，边界 token 必须保留。把被过滤项概率置零并重新归一化，最后使用 default_rng(seed).choice 采样。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `logits`：形状为 (vocab_size,) 的 float64 未归一化分数。
- `temperature`：大于 0 的温度。
- `top_k`：0 或不超过词表大小的候选数。
- `top_p`：位于 (0,1] 的核采样阈值。
- `seed`：单次采样的整数随机种子。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(token_id, probabilities)`；token_id 为 Python int，probabilities 形状为 `(vocab_size,)`、总和为 1。

## 数学定义

$p_i=softmax(z_i/T)$。Top-P 选择按概率降序的最短前缀 S，使 $\sum_{i\in S}p_i\ge p$，过滤后再次归一化。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖不同词表、温度、Top-K 开关、Top-P 边界、极端 logits、概率并列和固定种子复现。本题误差标准为 `rtol=1e-10`、`atol=1e-12`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
