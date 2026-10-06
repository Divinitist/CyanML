# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：手写线性层前向及对输入、权重、偏置的完整反向传播。
# 语法：矩阵转置配合 @ 对齐链式法则维度；np.sum(axis=0) 汇总 batch 的偏置梯度。

def linear_forward_backward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    grad_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        x: 形状为 (batch_size,in_features) 的 float64 输入。
        weight: 形状为 (in_features,out_features) 的权重。
        bias: 形状为 (out_features,) 的偏置。
        grad_output: 形状为 (batch_size,out_features) 的上游梯度。

    Returns:
        返回 `(output, grad_x, grad_weight, grad_bias)`，形状依次为 `(N,O)`、`(N,I)`、`(I,O)`、`(O,)`。
    """
    # 前向矩阵乘法把 (N, I) 映射为 (N, O)，bias 沿 batch 维广播。
    output = x @ weight + bias
    # 对输入求导时乘 W^T，把上游梯度从输出维映射回输入维。
    grad_x = grad_output @ weight.T
    # x.T 汇总所有样本对每个 (输入, 输出) 权重的梯度贡献。
    grad_weight = x.T @ grad_output
    # bias 在前向被每个样本共享，因此其梯度要沿 batch 轴求和。
    grad_bias = np.sum(grad_output, axis=0)
    # 按前向结果、输入梯度、权重梯度、偏置梯度的接口顺序返回。
    return output, grad_x, grad_weight, grad_bias
