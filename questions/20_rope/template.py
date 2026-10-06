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
    pass
