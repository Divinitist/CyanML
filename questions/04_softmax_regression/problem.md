# Softmax 多分类回归

## 背景

Softmax 回归是逻辑回归的多分类推广。模型为每个类别产生一个 logit，通过 Softmax 得到概率，并使用平均交叉熵训练。

## 需求

实现 `softmax_regression_step(x, targets, weights, bias, learning_rate, l2)`，计算更新前的概率、损失、预测和梯度，并执行一步梯度下降。

## 输入

- `x`：形状 `(batch_size, num_features)` 的 `float64` 数组；
- `targets`：形状 `(batch_size,)` 的整数类别下标；
- `weights`：形状 `(num_features, num_classes)` 的 `float64` 数组；
- `bias`：形状 `(num_classes,)` 的 `float64` 数组；
- `learning_rate`、`l2`：Python 浮点数。

## 输出

返回八元组：`probabilities`、`loss`、`predictions`、`grad_weights`、`grad_bias`、`new_weights`、`new_bias`。其中 L2 损失为 $\frac{\lambda}{2}\sum W^2$，不正则化偏置。

## 测试约定

100 组合法输入覆盖二分类到多分类、单样本、多 batch、零参数、大幅值 logits、不同学习率与正则强度。误差为 `rtol=1e-10`、`atol=1e-12`。
