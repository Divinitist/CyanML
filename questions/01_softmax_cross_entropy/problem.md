# Softmax 与交叉熵

## 背景

多分类模型通常输出每个类别的 logits。Softmax 将 logits 转换为概率分布，交叉熵衡量该分布与真实类别之间的差异。本题要求一次完成数值稳定的概率计算和批量平均损失计算。

## 需求

实现 `softmax_cross_entropy(logits, targets)`，返回 softmax 概率和批量平均交叉熵损失。

## 输入

- `logits`：形状为 `(batch_size, num_classes)` 的 `float64` NumPy 数组。
- `targets`：形状为 `(batch_size,)` 的整数 NumPy 数组；每个元素都是合法的类别下标。

所有测试输入均合法，不需要额外进行参数校验。

## 输出

返回二元组 `(probabilities, loss)`：

- `probabilities`：形状与 `logits` 相同的 `float64` 数组，每行都是概率分布。
- `loss`：Python 浮点数，表示整个 batch 的平均交叉熵。

## 数学定义

对于第 `i` 个样本和第 `j` 个类别：

$$p_{ij}=\frac{e^{z_{ij}}}{\sum_k e^{z_{ik}}}$$

平均交叉熵为：

$$L=-\frac{1}{N}\sum_i \log p_{i,y_i}$$

## 测试约定

- 默认数据类型为 `float64`。
- 共运行 100 组输入，覆盖单样本、多样本、不同类别数、极大与极小 logits、整体平移等情况。
- 本题误差标准为 `rtol=1e-10`、`atol=1e-12`。
