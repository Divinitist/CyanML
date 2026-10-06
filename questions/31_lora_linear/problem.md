# LoRA Linear

## 背景

LoRA 冻结原始线性层，仅训练两个低秩矩阵来表示权重增量，显著减少微调参数。缩放和 dropout 的位置会影响与常用实现的一致性。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `lora_linear`。计算基础分支 x@weight+bias，以及 LoRA 分支 dropout(x)@lora_a@lora_b*(alpha/rank)。dropout 只作用于 LoRA 分支：training=False 或 dropout_p=0 时直接使用 x；训练时用 default_rng(seed) 生成保留掩码并采用 inverted dropout。bias=None 时不加偏置。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：形状为 (...,in_features) 的 float64 输入。
- `weight`：形状为 (in_features,out_features) 的冻结权重。
- `bias`：None 或形状为 (out_features,) 的偏置。
- `lora_a / lora_b`：形状为 (in_features,rank) 和 (rank,out_features) 的低秩矩阵。
- `alpha`：LoRA 缩放超参数，实际倍率为 alpha/rank。
- `dropout_p / training / seed`：丢弃概率、训练标志和可复现随机种子。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回前导维度与 x 相同、末维为 out_features 的 float64 数组。

## 数学定义

$y=xW+b+\frac{\alpha}{r}Dropout(x)AB$；训练时 $Dropout(x)=m\odot x/(1-p)$，其中 $m\sim Bernoulli(1-p)$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖不同秩和缩放、带/不带偏置、训练/推理、零及非零 dropout、二维/三维输入和固定种子。本题误差标准为 `rtol=1e-9`、`atol=1e-11`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
