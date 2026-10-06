# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：完成线性回归的一次前向、MSE 求导和梯度下降更新。
# 语法：@ 是矩阵乘法；x.T 转置特征矩阵，使其可一次汇总整批权重梯度。

def linear_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), dtype float64.
        weights: Shape (num_features,), dtype float64.
        bias: Scalar bias before the update.
        learning_rate: Scalar gradient-descent learning rate.

    Returns:
        predictions, loss, grad_weights, grad_bias, new_weights, new_bias.

    The derivative of mean squared error contributes the factor 2 / N.
    Gradients are computed from the original parameters; the update is applied
    only after both gradients have been obtained.
    """
    # @ 将每个样本与权重做点积；标量 bias 会广播到整批样本。
    predictions = x @ weights + bias
    # 计算预测值减真实值的残差；其符号决定梯度更新方向。
    residuals = predictions - targets
    # 对残差平方取批平均，得到均方误差。 `np.mean` 沿指定轴求算术平均；`float(...)` 把 NumPy 标量转换为公开接口要求的 Python 浮点数。
    loss = float(np.mean(residuals**2))
    # MSE 对预测值的导数为 2 * residual / batch_size。
    grad_predictions = 2.0 * residuals / x.shape[0]
    # x.T 把每个样本的梯度汇总到对应特征权重上。
    grad_weights = x.T @ grad_predictions
    # 偏置被整批样本共享，因此用 `np.sum` 汇总所有预测梯度；`float` 把 NumPy 标量转成 Python 标量。
    grad_bias = float(np.sum(grad_predictions))
    # 两个参数都使用更新前算出的同一组梯度做梯度下降。
    new_weights = weights - learning_rate * grad_weights
    # 从旧偏置减去学习率乘梯度，完成偏置的一次梯度下降更新。 `float(...)` 把 NumPy 标量转换为公开接口要求的 Python 浮点数。
    new_bias = float(bias - learning_rate * grad_bias)
    # 返回更新前的前向量、梯度以及更新后的参数，便于逐项核对一次训练步骤。
    return predictions, loss, grad_weights, grad_bias, new_weights, new_bias
