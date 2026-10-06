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
    # 这个玩意要做的就是，把每个 kv 头存下来
    l, d = x.shape
    qn = num_query_heads
    kn = num_kv_heads
    h = d // qn # 按分组注意力，kv 头肯定比 q 头更少，所以 h 用 q 头数量来定义
    q = (x @ w_q).reshape((l, qn, h)).transpose(1, 0, 2)
    k = (x @ w_k).reshape((l, kn, h)).transpose(1, 0, 2)
    v = (x @ w_v).reshape((l, kn, h)).transpose(1, 0, 2)
    # 拼接 seq 维
    if cache_k is None:
        new_cache_k, new_cache_v, past = k, v, 0
    else:
        past = cache_k.shape[1]
        new_cache_k = np.concatenate((cache_k, k), axis=1)
        new_cache_v = np.concatenate((cache_v, v), axis=1)
    mask = past + np.arange(l)[:, None] >= np.arange(new_cache_k.shape[1])
    scores = np.where(mask[None], (q @ new_cache_k.transpose(0, 2, 1)) / np.sqrt(h), -np.inf)
    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)
    attn = (softmax(scores) @ new_cache_v).transpose(1, 0, 2).reshape((l, d))
    return attn @ w_o, new_cache_k, new_cache_v
    
