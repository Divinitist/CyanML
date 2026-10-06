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
    pass
