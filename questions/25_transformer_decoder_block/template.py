import numpy as np


def transformer_decoder_block(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    position_ids: np.ndarray,
) -> np.ndarray:
    """
    Args:
        x: 形状为 (sequence_length, model_dim) 的 float64 隐状态。
        params: 注意力、SwiGLU 和两组 RMSNorm 参数字典；投影不含偏置。
        num_heads: 能整除模型维度的头数，头维为偶数。
        position_ids: 形状为 (L,) 的非负绝对位置。

    Returns:
        返回形状为 `(L, model_dim)` 的解码器块输出。
    """
    pass
