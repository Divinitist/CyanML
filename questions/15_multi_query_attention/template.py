import numpy as np


def multi_query_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_query_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (L,D) 的 float64 输入。
            w_q: 形状为 (D,D)。
            w_k / w_v: 形状为 (D, head_dim)。
            w_o: 形状为 (D,D)。
            num_query_heads: 查询头数，head_dim=D/H。
            mask: None 或形状为 (L,L) 的布尔掩码。

        Returns:
            返回形状为 `(L,D)` 的 float64 数组。
        """
    pass
