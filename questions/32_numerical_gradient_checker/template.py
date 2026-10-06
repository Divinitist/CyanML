import numpy as np


def check_gradient(
    function: callable,
    x: np.ndarray,
    analytic_gradient: np.ndarray,
    epsilon: float,
    tolerance: float,
) -> tuple[np.ndarray, float, bool]:
    """
    Args:
        function: 接收一个与 x 同形数组并返回标量的纯函数。
        x: 任意形状的小型 float64 数组。
        analytic_gradient: 与 x 同形的解析梯度。
        epsilon: 中心差分的正扰动步长。
        tolerance: 最大相对误差阈值。

    Returns:
        返回 `(numerical_gradient, max_relative_error, passed)`；梯度与 x 同形，误差为 Python float，passed 为 Python bool。
    """
    pass
