# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：沿时间反向递推 GAE 优势，并由 value baseline 还原训练用 returns。
# API：np.empty 只分配未初始化缓冲区，因此循环必须为每个位置都写入结果。

def generalized_advantage_estimation(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the GAE recurrence backward because A_t depends on A_(t+1).

    The same non-terminal mask gates both value bootstrapping in the TD
    residual and propagation of the following advantage. This also supports
    several episodes packed into one one-dimensional rollout.
    """
    # empty 只分配缓冲区；下方反向循环会覆盖每个时间位置。
    advantages = np.empty(rewards.shape, dtype=np.float64)
    # 保存反向递推中的下一时刻 GAE；末端从 0 开始。
    next_advantage = 0.0
    # A_t 依赖 A_(t+1)，所以必须从 rollout 末端向前递推。
    for index in range(rewards.size - 1, -1, -1):
        # episode 结束处 mask 为 0，同时截断价值 bootstrap 和优势传播。
        non_terminal = 1.0 - float(dones[index])
        # 一步 TD 残差：即时奖励 + 折扣后的下一状态价值 - 当前价值。
        delta = (
            rewards[index]
            + gamma * non_terminal * values[index + 1]
            - values[index]
        )
        # lambda 控制后续 TD 残差向当前时刻传播的衰减程度。
        next_advantage = (
            delta + gamma * gae_lambda * non_terminal * next_advantage
        )
        # 把当前时刻递推得到的 GAE 写入对应时间位置。
        advantages[index] = next_advantage
    # 训练 value function 的目标等于优势加回当前 baseline。
    returns = advantages + values[:-1]
    # 同时返回策略优势和用于训练价值函数的回报目标。
    return advantages, returns
