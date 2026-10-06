# Decoder-only Transformer

## 背景

Decoder-only Transformer 把 token embedding、若干因果解码器块、最终归一化和词表投影连接成完整前向过程。本题综合前面的 RMSNorm、RoPE、注意力和 SwiGLU。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `decoder_only_transformer`。输入是单条序列。每个 blocks 元素包含 w_q/w_k/w_v/w_o、w_gate/w_up/w_down、norm1_scale/norm2_scale；所有线性层无偏置。每层使用 Pre-RMSNorm、对 Q/K 全头维施加 RoPE、下三角因果掩码、注意力残差、SwiGLU 残差。最后做 RMSNorm，并与 embedding_weight 共享 LM Head 权重。epsilon=1e-6，位置从 0 开始。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `token_ids`：形状为 (sequence_length,) 的整数 token 序列。
- `embedding_weight`：形状为 (vocab_size,model_dim) 的输入/输出共享权重。
- `blocks`：一个或多个解码器块参数字典组成的列表。
- `final_norm_scale`：形状为 (model_dim,) 的最终 RMSNorm 缩放。
- `num_heads`：能整除 model_dim 且头维为偶数的头数。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回形状为 `(sequence_length,vocab_size)` 的 next-token logits。

## 数学定义

每层为 $h'=h+Attn(RMSNorm(h))$、$h''=h'+SwiGLU(RMSNorm(h'))$；最终 $logits=RMSNorm(h'')W_{emb}^T$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖不同词表/序列长度、一至两层、不同头数和偶数头维、不同 FFN 宽度、重复 token 与单 token 序列。本题误差标准为 `rtol=1e-8`、`atol=1e-10`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
