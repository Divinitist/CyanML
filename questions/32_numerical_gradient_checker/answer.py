# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：用中心差分逐元素估计数值梯度，并与解析梯度比较最大相对误差。
# API：np.ndindex(shape) 遍历任意维数组的索引元组；empty_like 创建同形结果缓冲区。

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
    # 在副本上逐元素扰动，避免修改调用方传入的 x。
    working = x.copy()
    # 分配与输入同形的 float64 数值梯度数组，循环会填满每个元素。 `np.empty_like` 只按参照数组的形状和 dtype 分配内存，不会初始化元素。
    numerical = np.empty_like(working, dtype=np.float64)
    # ndindex 为任意维数组依次产生完整索引元组。
    for index in np.ndindex(working.shape):
        # 暂存当前元素原值，以便正负扰动后恢复输入。
        original = working[index]
        # 中心差分用 f(x+eps) 与 f(x-eps)，截断误差小于单边差分。
        working[index] = original + epsilon
        # 在当前维度加 epsilon 后求函数值，作为中心差分上端点。 `float(...)` 把 NumPy 标量转换为公开接口要求的 Python 浮点数。
        plus = float(function(working))
        # 把当前元素改为 `original-epsilon`，在同一坐标取得中心差分的下端点。
        working[index] = original - epsilon
        # 在当前维度减 epsilon 后求函数值，作为中心差分下端点。 `float(...)` 把 NumPy 标量转换为公开接口要求的 Python 浮点数。
        minus = float(function(working))
        # 每次计算后恢复原值，防止不同维度的扰动互相污染。
        working[index] = original
        # 分配与输入同形的 float64 数值梯度数组，循环会填满每个元素。
        numerical[index] = (plus - minus) / (2.0 * epsilon)
    # 分母至少为 1，避免接近零的梯度把微小绝对误差无限放大。
    denominator = np.maximum(1.0, np.maximum(np.abs(numerical), np.abs(analytic_gradient)))
    # 取所有元素相对误差的最大值，作为梯度检查的单一判据。 `float(...)` 把 NumPy 标量转换为公开接口要求的 Python 浮点数。
    max_error = float(np.max(np.abs(numerical - analytic_gradient) / denominator))
    # 返回数值梯度、最大相对误差，以及误差是否落在容差内。
    return numerical, max_error, bool(max_error <= tolerance)
