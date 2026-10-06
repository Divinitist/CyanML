# 线性回归

## 背景

线性回归使用特征的加权和预测连续值。本题把一次完整训练步骤放在一个公开接口中，用于理解预测、均方误差、参数梯度和梯度下降更新之间的关系。

## 需求

实现 `linear_regression_step(x, targets, weights, bias, learning_rate)`。计算更新前的预测与平均均方误差，并返回参数梯度和完成一步梯度下降后的参数。

## 输入

- `x`：形状 `(batch_size, num_features)` 的 `float64` 数组。
- `targets`：形状 `(batch_size,)` 的 `float64` 数组。
- `weights`：形状 `(num_features,)` 的 `float64` 数组。
- `bias`：Python 浮点数。
- `learning_rate`：正的 Python 浮点数。

测试输入均合法。

## 输出

返回六元组：

1. `predictions`：更新前的预测，形状 `(batch_size,)`；
2. `loss`：更新前的平均均方误差；
3. `grad_weights`：权重梯度，形状 `(num_features,)`；
4. `grad_bias`：偏置梯度；
5. `new_weights`：一步更新后的权重；
6. `new_bias`：一步更新后的偏置。

## 数学定义

$$\hat y=Xw+b$$

$$L=\frac{1}{N}\sum_i(\hat y_i-y_i)^2$$

## 测试约定

共 100 组 `float64` 输入，覆盖不同 batch、特征数、零学习率尺度、零残差、不同数值尺度和固定随机综合输入。本题误差标准为 `rtol=1e-10`、`atol=1e-12`。
