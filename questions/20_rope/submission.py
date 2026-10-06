import numpy as np


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    position_ids: np.ndarray,
    base: float = 10000.0,
    rotary_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Args:
            q / k: 形状为 (sequence_length, num_heads, head_dim) 的 float64 数组。
            position_ids: 形状为 (sequence_length,) 的位置数组。
            base: RoPE 频率基数。
            rotary_dim: None 或不超过 head_dim 的正偶数。

        Returns:
            返回旋转后的 `(q_rotated, k_rotated)`，两者形状不变且不修改输入。
        """
    l, n, h = q.shape
    h = rotary_dim if rotary_dim else h
    frequencies = base ** (-np.arange(0, h, 2) / h)
    # 这里的 position_ids 必须扩成 (seq, 1)，而(seq,) 是一维，逗号是为了避免和运算括号弄混
    angles = position_ids[:, None] * frequencies[None,:]
    sine = np.sin(angles)[:,None,:]
    cosine = np.cos(angles)[:,None,:]

    def rotate(x):
        ret = x.copy()
        x2i = x[..., :h:2]
        x2i1 = x[..., 1:h:2]
        ret[..., :h:2] = x2i * cosine - x2i1 * sine
        ret[..., 1:h:2] = x2i * sine + x2i1 * cosine
        return ret
    return rotate(q), rotate(k)
