# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：完成二分类逻辑回归的一次稳定前向、反向和参数更新。
# API：np.logaddexp(a, b) 稳定计算 log(exp(a)+exp(b))，避免直接 sigmoid/log 溢出。

def logistic_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), binary float64 labels.
        weights: Shape (num_features,), dtype float64.
        bias: Scalar bias.
        learning_rate: Scalar update rate.
        l2: L2 regularization coefficient.

    Returns:
        probabilities, loss, predictions, grad_weights, grad_bias,
        new_weights, new_bias.

    The two sigmoid branches prevent overflow for logits with large magnitude.
    logaddexp computes binary cross-entropy directly from logits, avoiding
    log(0). L2 applies to weights but not to the bias.
    """
    # 先得到未经过 Sigmoid 的线性分数。
    logits = x @ weights + bias
    # 为 Sigmoid 概率分配与 logits 同形、同 dtype 的缓冲区；`np.empty_like` 不初始化元素，下面两个互补分支必须把它填满。
    probabilities = np.empty_like(logits)
    # 正负区间采用不同等价公式，避免 exp 遇到大数时溢出。
    positive = logits >= 0
    # 只给当前数值区间对应的位置写入稳定的 Sigmoid 结果，两个互补掩码最终覆盖全部元素。 `np.exp` 对数组逐元素取自然指数。
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    # 仅对负 logit 子集取指数，使用不会溢出的 Sigmoid 等价式。 `np.exp` 对数组逐元素取自然指数。
    exp_logits = np.exp(logits[~positive])
    # 只给当前数值区间对应的位置写入稳定的 Sigmoid 结果，两个互补掩码最终覆盖全部元素。
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)

    # logaddexp(0, z) 稳定计算 log(1 + exp(z))，无需对概率取 log。
    data_loss = np.mean(np.logaddexp(0.0, logits) - targets * logits)
    # L2 只约束权重；0.5 让求导后的系数恰好是 l2 * weights。
    loss = float(data_loss + 0.5 * l2 * np.sum(weights**2))
    # 把概率转换成离散预测类别。 `astype` 逐元素转换 dtype，布尔值会变成 0/1 整数。
    predictions = (probabilities >= 0.5).astype(np.int64)
    # 计算损失对 logits 的批平均梯度，作为线性层反传入口。
    grad_logits = (probabilities - targets) / x.shape[0]
    # 线性层反传后再叠加 L2 梯度，偏置不参与正则化。
    grad_weights = x.T @ grad_logits + l2 * weights
    # 偏置作用于整批 logits，因此用 `np.sum` 汇总所有 logit 梯度；`float` 转成 Python 标量。
    grad_bias = float(np.sum(grad_logits))
    # 从旧权重减去学习率乘梯度，完成权重的一次梯度下降更新。
    new_weights = weights - learning_rate * grad_weights
    # 从旧偏置减去学习率乘梯度，完成偏置的一次梯度下降更新。 `float(...)` 把 NumPy 标量转换为公开接口要求的 Python 浮点数。
    new_bias = float(bias - learning_rate * grad_bias)
    # 返回概率、损失、分类结果、两项梯度和更新后的参数，供评测逐项比较。
    return probabilities, loss, predictions, grad_weights, grad_bias, new_weights, new_bias
