import numpy as np


def apply_scaled_rope(
    q: np.ndarray,
    k: np.ndarray,
    position_ids: np.ndarray,
    scale: float,
    method: str,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Args:
            q / k: 形状为 (L,H,d) 的 float64 数组。
            position_ids: 形状为 (L,) 的绝对位置。
            scale: 大于等于 1 的上下文扩展倍率。
            method: 字符串 linear 或 ntk。
            base: 原始 RoPE 基数。

        Returns:
            返回缩放旋转后的 Q/K 二元组，形状与输入一致。
        """
    pass
