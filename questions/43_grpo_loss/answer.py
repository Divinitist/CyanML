# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：组合组内奖励标准化、PPO 式裁剪代理目标和 reference KL 惩罚。
# API：advantages[..., None] 新增 token 轴用于广播；布尔 mask 转 float 后可参与加权求和。

def grpo_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    rewards: np.ndarray,
    mask: np.ndarray,
    clip_epsilon: float,
    kl_beta: float,
    normalization_epsilon: float = 1e-8,
) -> float:
    """
    Build response-level relative advantages and broadcast them over tokens.

    Token objectives are masked and normalized per response before the outer
    mean, so longer answers do not receive more weight solely due to length.
    The exponential KL estimator is non-negative and equals zero when the
    current and reference token log-probabilities match.
    """
    # 每个 prompt 的候选响应组成一组，先减组均值得到相对奖励。
    centered_rewards = rewards - np.mean(rewards, axis=1, keepdims=True)
    # 使用组内方差标准化，使不同 prompt 的奖励尺度可比较。
    variance = np.mean(centered_rewards**2, axis=1, keepdims=True)
    # 把奖励或 TD 残差转换为用于策略更新的相对优势。 `np.sqrt` 对数组逐元素开平方。
    advantages = centered_rewards / np.sqrt(variance + normalization_epsilon)

    # token 级新旧策略概率比用于构造 PPO 式代理目标。
    ratio = np.exp(new_log_probs - old_log_probs)
    # 把策略概率比限制在 `[1-epsilon, 1+epsilon]`，抑制过大更新。 `np.clip` 把每个元素截断到给定闭区间。
    clipped_ratio = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    # advantages[..., None] 新增 token 轴，把响应级优势广播到每个 token。
    surrogate = np.minimum(
        ratio * advantages[..., None],
        clipped_ratio * advantages[..., None],
    )

    # 该指数形式是非负 KL 估计，策略与参考相同时严格为零。
    reference_ratio_log = reference_log_probs - new_log_probs
    # 用 `exp(r)-r-1` 构造非负的逐 token KL 估计。 `np.exp` 对数组逐元素取自然指数。
    kl_estimate = np.exp(reference_ratio_log) - reference_ratio_log - 1.0
    # 从策略收益中扣除参考模型 KL 惩罚，约束策略不要漂移过远。
    token_objective = surrogate - kl_beta * kl_estimate

    # mask 排除 padding；先对每个响应求有效 token 均值，避免长响应权重更大。
    valid = mask.astype(np.float64)
    # 只对 mask 标记的有效 token 求平均，使响应长度不改变样本权重。 `np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴。
    response_objective = np.sum(token_objective * valid, axis=-1) / np.sum(
        valid, axis=-1
    )
    # 最外层平均所有组和响应，并取负号转成需要最小化的 loss。
    return float(-np.mean(response_objective))
