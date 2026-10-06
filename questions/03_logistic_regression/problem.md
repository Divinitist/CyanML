# 逻辑回归

## 背景

逻辑回归将线性输出转换为二分类概率。本题覆盖概率、二元交叉熵、分类结果、L2 正则化梯度以及一步梯度下降更新。

## 需求

实现 `logistic_regression_step(x, targets, weights, bias, learning_rate, l2)`，所有损失和梯度都基于更新前的参数计算。

## 输入

- `x`：`(batch_size, num_features)` 的 `float64` 数组；
- `targets`：`(batch_size,)` 的 `float64` 数组，元素为 0 或 1；
- `weights`：`(num_features,)` 的 `float64` 数组；
- `bias`、`learning_rate`、`l2`：Python 浮点数。

## 输出

返回七元组 `(probabilities, loss, predictions, grad_weights, grad_bias, new_weights, new_bias)`。`predictions` 使用 `0.5` 作为分类阈值，数据类型为整数。损失中的 L2 项定义为：

$$\frac{\lambda}{2}\sum_j w_j^2$$

## 测试约定

100 组合法 `float64` 输入覆盖单样本、多特征、零权重、零正则、不同正则强度和大幅值 logits。误差为 `rtol=1e-10`、`atol=1e-12`。
