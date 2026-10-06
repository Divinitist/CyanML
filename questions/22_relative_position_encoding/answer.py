# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：生成可直接加到注意力分数上的 ALiBi 或查表式相对位置偏置。
# API：np.clip 把距离限制到表范围；table[:, indices] 用整数矩阵同时索引每个头的位置桶。

def relative_position_bias(
    query_length: int,
    key_length: int,
    num_heads: int,
    mode: str,
    max_distance: int,
    table: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            query_length / key_length: 查询和键的正整数长度，可以不同。
            num_heads: 注意力头数。
            mode: 字符串 alibi 或 table。
            max_distance: table 模式的最大有符号距离。
            table: table 模式下形状为 (H,2*max_distance+1)；alibi 时为 None。

        Returns:
            返回形状为 `(num_heads, query_length, key_length)` 的 float64 偏置，可直接加到注意力分数。
        """
    # 列式 query 位置减行式 key 位置，通过广播得到完整相对距离矩阵。
    query_positions = np.arange(query_length)[:, None]
    # `np.arange(key_length)[None, :]` 建立形状 (1,K) 的 Key 位置行向量，供二维广播。
    key_positions = np.arange(key_length)[None, :]
    # 行向量 Key 位置减列向量 Query 位置，通过广播得到形状 (Q,K) 的有符号相对距离。
    relative = key_positions - query_positions
    # ALiBi 模式直接按距离公式生成偏置；否则走下面的可学习查表分支。
    if mode == "alibi":
        # 不同头使用不同斜率，对绝对距离施加线性负惩罚。
        slopes = 2.0 ** (-8.0 * (np.arange(num_heads) + 1) / num_heads)
        # 负的绝对距离乘每头斜率，得到 ALiBi 的线性距离惩罚。
        return -slopes[:, None, None] * np.abs(relative)[None, :, :]
    # 超出范围的距离压到两端桶，再平移成从零开始的整数索引。
    indices = np.clip(relative, -max_distance, max_distance) + max_distance
    # table[:, indices] 为每个头按同一索引矩阵查表，结果为 (H, Q, K)。
    return table[:, indices]
