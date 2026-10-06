# Linear 完整 backward

## 背景

全连接层的反向传播是理解自动微分的基础。一个上游梯度需要同时沿计算图传播到输入、权重和偏置，并对批次维正确求和。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `linear_forward_backward`。前向计算 output=x@weight+bias，其中 x 为二维批次矩阵。给定与 output 同形的 grad_output，使用矩阵微分公式一次性计算 grad_x、grad_weight 和 grad_bias。函数只返回计算结果，不在原数组上累积梯度。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：形状为 (batch_size,in_features) 的 float64 输入。
- `weight`：形状为 (in_features,out_features) 的权重。
- `bias`：形状为 (out_features,) 的偏置。
- `grad_output`：形状为 (batch_size,out_features) 的上游梯度。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(output, grad_x, grad_weight, grad_bias)`，形状依次为 `(N,O)`、`(N,I)`、`(I,O)`、`(O,)`。

## 数学定义

$Y=XW+b$，$\partial L/\partial X=G W^T$，$\partial L/\partial W=X^T G$，$\partial L/\partial b=\sum_nG_n$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖单/多样本、不同输入输出维度、零值、负值、不同数值尺度，并通过独立数值差分覆盖所有梯度方向。本题误差标准为 `rtol=1e-10`、`atol=1e-12`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
