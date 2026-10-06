import numpy as np


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
    o = x @ weight + bias
    # aL/ax = aL/ao * ao/ax = aL/ao * w
    grad_x = grad_output @ weight.T
    # aL/aw = aL/ao * ao/aw = aL/ao * x
    grad_w = x.T @ grad_output
    # aL/ab = aL/ao * ao/ab = aL/ao * 1
    grad_b = np.sum(grad_output.T, axis=-1)
    return o, grad_x, grad_w, grad_b
