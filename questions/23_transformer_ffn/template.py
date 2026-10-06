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
    pass
