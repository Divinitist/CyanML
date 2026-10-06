# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：更新 Adam 的一、二阶矩，并对参数执行带解耦权重衰减的 AdamW 步骤。
# API：np.sqrt 逐元素开平方；数组的 ** 和四则运算均按元素执行并支持广播。

def adamw_update(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        parameter: Parameter values before the update.
        gradient: Current gradient, same shape as parameter.
        first_moment: First-moment state before the update.
        second_moment: Second-moment state before the update.
        step: One-based update number.
        learning_rate: Optimizer learning rate.
        beta1: First-moment decay coefficient.
        beta2: Second-moment decay coefficient.
        epsilon: Denominator stabilizer.
        weight_decay: Decoupled weight-decay coefficient.

    Returns:
        new_parameter, new_first_moment, new_second_moment.

    Bias correction uses the current one-based step. AdamW decay multiplies
    the original parameter directly and never enters either moment estimate.
    """
    # 指数滑动平均分别估计梯度的一阶矩和平方梯度的二阶矩。
    new_first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
    # 用平方梯度更新 Adam 的二阶矩指数滑动平均。
    new_second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
    # 训练初期矩估计偏向零，用当前的一基 step 做偏差修正。
    corrected_first = new_first_moment / (1.0 - beta1**step)
    # 除以 `1-beta2**step`，修正训练初期二阶矩偏向零的问题。
    corrected_second = new_second_moment / (1.0 - beta2**step)
    # AdamW 的权重衰减直接作用于参数，不混入梯度或矩状态。
    decayed_parameter = parameter * (1.0 - learning_rate * weight_decay)
    # sqrt 把二阶矩变回梯度量纲，epsilon 防止分母为零。
    new_parameter = decayed_parameter - learning_rate * corrected_first / (
        np.sqrt(corrected_second) + epsilon
    )
    # 返回更新后的参数以及下一优化步骤需要继续保存的一阶、二阶矩状态。
    return new_parameter, new_first_moment, new_second_moment
