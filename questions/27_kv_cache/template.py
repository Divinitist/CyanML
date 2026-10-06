import numpy as np


def kv_cache_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    cache_k: np.ndarray | None = None,
    cache_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        x: 当前分块，形状为 (chunk_length, model_dim)。
        w_q: 形状为 (D,D) 的查询投影。
        w_k / w_v: 形状为 (D,num_kv_heads*head_dim) 的 KV 投影。
        w_o: 形状为 (D,D) 的输出投影。
        num_query_heads / num_kv_heads: 查询头数与 KV 头数，前者可被后者整除。
        cache_k / cache_v: None 或形状为 (num_kv_heads,past_length,head_dim) 的历史缓存。

    Returns:
        返回 `(output, new_cache_k, new_cache_v)`；output 为 `(chunk_length,D)`，新缓存长度为 past_length+chunk_length。
    """
    pass
