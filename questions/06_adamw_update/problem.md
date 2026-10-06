# AdamW 更新

## 背景

AdamW 使用梯度的一阶矩和二阶矩调整每个参数的更新幅度，并将权重衰减与梯度更新解耦。本题实现一次完整的 AdamW 参数更新。

## 需求

实现 `adamw_update(parameter, gradient, first_moment, second_moment, step, learning_rate, beta1, beta2, epsilon, weight_decay)`。

## 输入

- 前四个数组形状完全相同，数据类型为 `float64`；
- `step` 是当前这一次更新的序号，从 1 开始；
- 其余参数均为合法的 Python 浮点数。

## 输出

返回 `(new_parameter, new_first_moment, new_second_moment)`，三个数组形状均与 `parameter` 相同。

一阶矩和二阶矩需要进行 bias correction。权重衰减采用 AdamW 的 decoupled weight decay，不加入梯度或矩估计。

## 测试约定

100 组输入覆盖首次更新、较大 step、零梯度、零矩、不同 beta、epsilon、权重衰减以及向量和矩阵参数。误差为 `rtol=1e-10`、`atol=1e-12`。
