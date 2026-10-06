import numpy as np


def cross_attention(
    query: np.ndarray,
    context: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            query: 形状为 (query_length, model_dim)。
            context: 形状为 (context_length, model_dim)。
            w_q / w_k / w_v / w_o: 形状均为 (model_dim, model_dim)。
            num_heads: 能整除模型维度的头数。
            mask: None 或形状为 (Q,K) 的布尔掩码。

        Returns:
            返回形状为 `(query_length, model_dim)` 的 float64 数组。
        """
    Q, D = query.shape
    C = context.shape[0]
    N = num_heads
    H = D // N
    q = (query @ w_q).reshape((Q, N, H)).transpose(1, 0, 2)
    k = (context @ w_k).reshape((C, N, H)).transpose(1, 0, 2)
    v = (context @ w_v).reshape((C, N, H)).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(H)
    if mask is not None:
        scores = np.where(mask[None,:,:], scores, -np.inf)
    def softmax(x):
        x = x - x.max(axis=-1,keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1,keepdims=True)
    weights = softmax(scores)
    attn = (weights @ v).transpose(1, 0, 2).reshape((Q, D))
    return attn @ w_o
