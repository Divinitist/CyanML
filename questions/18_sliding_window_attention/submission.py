import numpy as np


def sliding_window_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    window_size: int,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (L,D) 的 float64 序列。
            w_q / w_k / w_v / w_o: 四个 (D,D) 投影矩阵。
            num_heads: 能整除 D 的注意力头数。
            window_size: 正整数窗口宽度，包含当前位置。

        Returns:
            返回形状为 `(L,D)` 的局部因果注意力结果。
        """
    L, D = x.shape
    N = num_heads
    H = D // N
    q = (x @ w_q).reshape((L, N, H)).transpose(1, 0, 2)
    k = (x @ w_k).reshape((L, N, H)).transpose(1, 0, 2)
    v = (x @ w_v).reshape((L, N, H)).transpose(1, 0, 2)
    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(H)
    idx = np.arange(L)
    mask = ((idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < window_size))
    scores = np.where(mask[None, :, :], scores, -np.inf)
    def softmax(x):
        x = x - x.max(axis=-1,keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1,keepdims=True)
    weights = softmax(scores)
    attn = (weights @ v).transpose(1, 0, 2).reshape(x.shape)
    return attn @ w_o
