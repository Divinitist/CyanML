# Encoder-Decoder Block

## 背景

序列到序列 Transformer 的解码器既要读取已生成的目标前缀，也要读取编码器表示。因此一个块包含因果自注意力、交叉注意力和前馈网络三个子层。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `encoder_decoder_block`。实现三个 Pre-LayerNorm 子层及各自残差。params 用 self_q/self_k/self_v/self_o 表示自注意力矩阵，cross_q/cross_k/cross_v/cross_o 表示交叉注意力矩阵，ff1/ff1_bias/ff2/ff2_bias 表示 GELU FFN，并提供 ln1/ln2/ln3 的 scale 和 bias。self_mask 与 cross_mask 中 True 表示可见，LayerNorm epsilon=1e-5。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `decoder_x`：形状为 (target_length, model_dim) 的解码器状态。
- `encoder_x`：形状为 (source_length, model_dim) 的编码器输出。
- `params`：三子层投影与 LayerNorm 参数字典。
- `num_heads`：能整除 model_dim 的头数。
- `self_mask`：None 或 (T,T) 布尔掩码。
- `cross_mask`：None 或 (T,S) 布尔掩码。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回形状为 `(target_length, model_dim)` 的更新后解码器状态。

## 数学定义

$y=x+SelfAttn(LN_1(x))$，$z=y+CrossAttn(LN_2(y),encoder)$，$o=z+FFN(LN_3(z))$；交叉注意力的 K/V 只来自 encoder_x。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖不同源/目标长度、多个头数、因果与随机交叉掩码、单 token 解码、不同 FFN 宽度和归一化参数。本题误差标准为 `rtol=1e-8`、`atol=1e-10`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
