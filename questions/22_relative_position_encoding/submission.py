import numpy as np


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
    relatives = np.arange(query_length)[:,None] - np.arange(key_length)[None,:]
    if mode == 'alibi':
        slopes = 2 ** (-8 * (np.arange(num_heads) + 1) / num_heads)
        # 简直就像写公式一样完美……
        return -slopes[:,None,None] * np.abs(relatives)[None,:,:]
    else:
        indices = np.clip(-relatives, -max_distance, max_distance) + max_distance
        return table[:, indices]
