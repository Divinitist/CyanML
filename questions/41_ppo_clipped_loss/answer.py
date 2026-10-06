# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：计算 PPO 的悲观裁剪代理目标，并返回需要最小化的负均值。
# API：np.clip 限制概率比率区间；np.minimum 对原始与裁剪目标逐元素取较小值。

def ppo_clipped_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float,
) -> float:
    """
    Form both the original and clipped surrogate objectives elementwise.

    Taking the minimum before negating is important: for negative advantages,
    the pessimistic branch is not necessarily the numerically smaller ratio.
    """
    # 对数概率作差再取 exp，得到新旧策略对同一动作的概率比。
    ratio = np.exp(new_log_probs - old_log_probs)
    # 限制比率偏离 1 的幅度，抑制单次策略更新过大。
    clipped_ratio = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    # 逐元素取较小代理目标，正负优势都选择更悲观的一支。
    surrogate = np.minimum(ratio * advantages, clipped_ratio * advantages)
    # 优化器最小化 loss，因此对需要最大化的代理目标取负均值。
    return float(-np.mean(surrogate))
