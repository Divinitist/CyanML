import numpy as np


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (sequence_length, model_dim) 的 float64 序列。
            w_q / w_k / w_v / w_o: 四个形状均为 (model_dim, model_dim) 的投影矩阵。
            num_heads: 注意力头数，能整除 model_dim。
            mask: None 或形状为 (L,L) 的布尔数组。

        Returns:
            返回形状为 `(L, model_dim)` 的注意力输出。
        """
    L, D = x.shape
    N = num_heads
    H = D // N
    q = (x @ w_q).reshape((L, N, H)).transpose(1, 0, 2)
    k = (x @ w_k).reshape((L, N, H)).transpose(1, 0, 2)
    v = (x @ w_v).reshape((L, N, H)).transpose(1, 0, 2)
    def softmax(x):
        xmax = x.max(axis=-1, keepdims=True)
        x = x - xmax
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(H)
    if mask is not None:
        scores = np.where(mask[None, :, :], scores, -np.inf)
    weights = softmax(scores)
    context = (weights @ v).transpose(1, 0, 2).reshape(x.shape)
    return context @ w_o
