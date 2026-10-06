# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：完成多分类 Softmax 回归的一次损失、梯度和参数更新。
# API：np.argmax(..., axis=1) 取每行最大类别；高级索引用于读取和修改每行目标项。

def softmax_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), integer class indices.
        weights: Shape (num_features, num_classes), dtype float64.
        bias: Shape (num_classes,), dtype float64.
        learning_rate: Scalar update rate.
        l2: L2 regularization coefficient.

    Returns:
        probabilities, loss, predictions, grad_weights, grad_bias,
        new_weights, new_bias.

    Row-wise maximum subtraction stabilizes exponentiation. The cross-entropy
    uses the same shift, and its logit gradient is probability minus one-hot
    target, averaged across the batch.
    """
    # 线性投影得到每个样本对各类别的原始分数。
    logits = x @ weights + bias
    # 逐行减最大值，保持 Softmax 不变并防止指数溢出。
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    # 把平移后的 logits 逐元素指数化，得到 Softmax 的未归一化权重。 `np.exp` 对数组逐元素取自然指数。
    exp_shifted = np.exp(shifted)
    # 每行指数值除以该行总和得到类别概率；`axis=1, keepdims=True` 产生 (N,1) 分母并广播到所有类别。
    probabilities = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    # 高级索引一次取出整批真实类别对应的分数。
    target_scores = shifted[np.arange(x.shape[0]), targets]
    # 对每行计算 `log(sum(exp(shifted)))-target_score`，再用 `np.mean` 得到批平均交叉熵。
    data_loss = np.mean(np.log(np.sum(exp_shifted, axis=1)) - target_scores)
    # 把交叉熵与 `0.5*l2*sum(weights**2)` 相加；`np.sum` 汇总全部权重平方，`float` 转成 Python 标量。
    loss = float(data_loss + 0.5 * l2 * np.sum(weights**2))
    # 把概率转换成离散预测类别。 `np.argmax` 返回指定轴上最大元素的下标。
    predictions = np.argmax(probabilities, axis=1)

    # Softmax 与交叉熵合并后的梯度是 probabilities - one_hot(targets)。
    grad_logits = probabilities.copy()
    # 在每行真实类别位置减 1，把概率矩阵原地变成 `softmax-one_hot` 梯度。
    grad_logits[np.arange(x.shape[0]), targets] -= 1.0
    # 除以 batch 大小，把样本梯度总量缩放为批平均梯度。
    grad_logits /= x.shape[0]
    # x.T 汇总批内样本贡献，L2 项只加到权重梯度。
    grad_weights = x.T @ grad_logits + l2 * weights
    # 沿 batch 汇总每个样本对共享偏置的梯度。 `np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴。
    grad_bias = np.sum(grad_logits, axis=0)
    # 从旧权重减去学习率乘梯度，完成权重的一次梯度下降更新。
    new_weights = weights - learning_rate * grad_weights
    # 从旧偏置减去学习率乘梯度，完成偏置的一次梯度下降更新。
    new_bias = bias - learning_rate * grad_bias
    # 返回 Softmax 前向结果、梯度和完成一次梯度下降后的参数。
    return probabilities, loss, predictions, grad_weights, grad_bias, new_weights, new_bias
