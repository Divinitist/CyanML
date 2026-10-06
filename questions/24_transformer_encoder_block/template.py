import numpy as np


def transformer_encoder_block(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    pre_norm: bool,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Args:
        x: 形状为 (sequence_length, model_dim) 的 float64 输入。
        params: 包含四个注意力矩阵、两层 FFN 参数和两组 LayerNorm 仿射参数的字典。
        num_heads: 能整除 model_dim 的注意力头数。
        pre_norm: True 选择 Pre-Norm，False 选择 Post-Norm。
        mask: None 或形状为 (L,L) 的布尔可见性掩码。

    Returns:
        返回形状与 x 相同的编码器块输出。
    """
    pass
