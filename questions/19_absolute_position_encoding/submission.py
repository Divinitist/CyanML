import numpy as np


def add_sinusoidal_position_encoding(
    x: np.ndarray,
    start_position: int = 0,
    base: float = 10000.0,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (sequence_length, model_dim) 的 float64 数组。
            start_position: 第一个 token 的非负绝对位置。
            base: 控制频率跨度的正浮点数。

        Returns:
            返回与 x 同形状的 `x + position_encoding`。
        """
    # theta[i] = base ^ {-2i/d}
    # pe[p][2i] = sin(p theta[i])
    # pe[p][2i+1] = cos(p theta[i])
    l, d = x.shape
    positions = start_position + np.arange(l)
    frequencies = base ** (-np.arange(0, d, 2) / d)
    # positions 在 seq 维，theta 在 emb 维，所以 positions 向 emb 维广播， theta 向 seq 维广播
    # 所谓广播，就是复制凑数。
    angles = positions[:, None] * frequencies[None, :]
    pe = np.zeros_like(x)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])
    return x + pe