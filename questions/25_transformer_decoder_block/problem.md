# Transformer Decoder Block

## 背景

现代 Decoder-only 模型通常组合因果自注意力、RoPE、RMSNorm 和 SwiGLU。本题把这些组件连接成一个可独立验证的 Pre-Norm 解码器块。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `transformer_decoder_block`。params 包含 w_q、w_k、w_v、w_o、w_gate、w_up、w_down、norm1_scale、norm2_scale。先 RMSNorm，再对投影后的全部 Q/K 头维应用 RoPE，使用下三角因果掩码；完成第一条残差后再次 RMSNorm，并通过 SwiGLU FFN 和第二条残差。RMSNorm epsilon=1e-6。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：形状为 (sequence_length, model_dim) 的 float64 隐状态。
- `params`：注意力、SwiGLU 和两组 RMSNorm 参数字典；投影不含偏置。
- `num_heads`：能整除模型维度的头数，头维为偶数。
- `position_ids`：形状为 (L,) 的非负绝对位置。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回形状为 `(L, model_dim)` 的解码器块输出。

## 数学定义

$RMSNorm(x)=x/\sqrt{mean(x^2)+10^{-6}}\odot g$；RoPE 旋转后使用因果注意力；$SwiGLU(x)=SiLU(xW_g)\odot(xW_u)$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖不同长度和起始位置、多个头数与偶数头维、因果边界、不同 FFN 宽度及 RMSNorm 缩放。本题误差标准为 `rtol=1e-8`、`atol=1e-10`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
