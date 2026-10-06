# 归一化层

## 背景

BatchNorm、LayerNorm 和 RMSNorm 使用不同的统计维度调整激活值。本题通过一个模式参数实现三种常见归一化的训练态前向计算。

## 需求

实现 `normalization(x, gamma, beta, mode, epsilon)`。

- `mode="batch"`：每个特征独立地在 batch 维计算均值和方差；
- `mode="layer"`：每个样本独立地在最后一维计算均值和方差；
- `mode="rms"`：每个样本在最后一维计算均方根，不减均值。

方差统一使用总体方差，即除以元素数量而不是 `N-1`。本题不维护 BatchNorm running statistics。

## 输入

- `x`：形状 `(batch_size, feature_size)` 的 `float64` 数组；
- `gamma`、`beta`：形状 `(feature_size,)` 的 `float64` 仿射参数；
- `mode`：上述三个字符串之一；
- `epsilon`：正浮点数。

## 输出

返回形状与 `x` 相同的 `float64` 数组。RMSNorm 在缩放后同样加上本题给定的 `beta`。

## 测试约定

100 组输入均匀覆盖三种模式、不同 batch 和特征数、常量输入、零输入、不同缩放与 epsilon。误差为 `rtol=1e-10`、`atol=1e-12`。
