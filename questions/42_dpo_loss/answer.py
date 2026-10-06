# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：比较当前策略与参考策略的偏好 margin，得到数值稳定的平均 DPO loss。
# API：np.logaddexp(0, -z) 等价于 softplus(-z)=-log(sigmoid(z))，大数时不会溢出。

def dpo_loss(
    chosen_log_probs: np.ndarray,
    rejected_log_probs: np.ndarray,
    reference_chosen_log_probs: np.ndarray,
    reference_rejected_log_probs: np.ndarray,
    beta: float,
) -> float:
    """
    Compare policy and reference preference margins.

    Since -log(sigmoid(z)) equals softplus(-z), logaddexp computes the loss
    without overflowing when the scaled preference margin is very large.
    """
    # margin 衡量模型对 chosen 相比 rejected 的相对偏好强度。
    policy_margin = chosen_log_probs - rejected_log_probs
    # 计算参考模型对 chosen 相比 rejected 的对数概率优势。
    reference_margin = reference_chosen_log_probs - reference_rejected_log_probs
    # 只奖励当前策略相对参考策略增加的偏好，beta 控制更新强度。
    logits = beta * (policy_margin - reference_margin)
    # logaddexp(0, -z) 稳定计算 -log(sigmoid(z))，避免 exp 溢出。
    return float(np.mean(np.logaddexp(0.0, -logits)))
