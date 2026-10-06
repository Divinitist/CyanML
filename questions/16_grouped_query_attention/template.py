import numpy as np


def grouped_query_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (L,D) 的 float64 输入。
            w_q: 形状为 (D,D) 的查询投影。
            w_k / w_v: 形状为 (D, num_kv_heads*head_dim)。
            w_o: 形状为 (D,D)。
            num_query_heads / num_kv_heads: 查询头数和 KV 头数。
            mask: None 或 (L,L) 布尔掩码。

        Returns:
            返回形状为 `(L,D)` 的注意力输出。
        """
    pass
