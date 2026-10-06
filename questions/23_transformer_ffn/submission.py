import numpy as np


def transformer_ffn(
    x: np.ndarray,
    w_up: np.ndarray,
    b_up: np.ndarray,
    w_down: np.ndarray,
    b_down: np.ndarray,
    activation: str,
    w_gate: np.ndarray | None = None,
    b_gate: np.ndarray | None = None,
) -> np.ndarray:
    """
    Args:
        x: 形状为 (..., model_dim) 的 float64 输入。
        w_up / b_up: 形状为 (model_dim, hidden_dim) 和 (hidden_dim,) 的内容投影。
        w_down / b_down: 形状为 (hidden_dim, model_dim) 和 (model_dim,) 的输出投影。
        activation: relu、gelu、glu、geglu 或 swiglu。
        w_gate / b_gate: 门控模式使用的 (model_dim, hidden_dim) 和 (hidden_dim,)；非门控模式为 None。

    Returns:
        返回与 x 前导维度相同、末维为 model_dim 的 float64 数组。
    """
    up = x @ w_up + b_up
    def gelu(value):
        return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * value ** 3)))
    if activation == 'relu':
        hidden = np.maximum(0, up)
    elif activation == 'gelu':
        hidden = gelu(up)
    else:
        gate = x @ w_gate + b_gate
        if activation == 'glu':
            gate = 1 / (1 + np.exp(-gate))
        elif activation == 'geglu':
            gate = gelu(gate)
        elif activation == 'swiglu':
            gate = gate / (1 + np.exp(-gate))
        hidden = up * gate
    return hidden @ w_down + b_down
        
