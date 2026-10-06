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
    pass
