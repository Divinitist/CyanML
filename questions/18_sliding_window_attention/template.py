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
    pass
