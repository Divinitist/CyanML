# Transformer Encoder Block

## 背景

编码器块把双向多头自注意力与逐位置前馈网络串联，并通过残差连接和 LayerNorm 稳定深层训练。Pre-Norm 与 Post-Norm 的运算顺序不同，不能混用。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `transformer_encoder_block`。实现单批次编码器块。params 必须使用键 w_q、w_k、w_v、w_o、w1、b1、w2、b2、ln1_scale、ln1_bias、ln2_scale、ln2_bias。注意力使用可选布尔 mask，FFN 使用 tanh 近似 GELU，LayerNorm 在末维计算且 epsilon=1e-5。pre_norm=True 时先归一化再进入子层；否则先残差相加再归一化。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：形状为 (sequence_length, model_dim) 的 float64 输入。
- `params`：包含四个注意力矩阵、两层 FFN 参数和两组 LayerNorm 仿射参数的字典。
- `num_heads`：能整除 model_dim 的注意力头数。
- `pre_norm`：True 选择 Pre-Norm，False 选择 Post-Norm。
- `mask`：None 或形状为 (L,L) 的布尔可见性掩码。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回形状与 x 相同的编码器块输出。

## 数学定义

$LN(x)=\gamma(x-\mu)/\sqrt{\sigma^2+10^{-5}}+\beta$。Pre-Norm 为 $y=x+Attn(LN_1(x))$、$z=y+FFN(LN_2(y))$；Post-Norm 为对应的残差后归一化。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖Pre/Post-Norm、不同头数和序列长度、双向与随机掩码、不同 FFN 宽度及归一化仿射参数。本题误差标准为 `rtol=1e-8`、`atol=1e-10`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
