import numpy as np


def encoder_decoder_block(
    decoder_x: np.ndarray,
    encoder_x: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    self_mask: np.ndarray | None = None,
    cross_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Args:
        decoder_x: 形状为 (target_length, model_dim) 的解码器状态。
        encoder_x: 形状为 (source_length, model_dim) 的编码器输出。
        params: 三子层投影与 LayerNorm 参数字典。
        num_heads: 能整除 model_dim 的头数。
        self_mask: None 或 (T,T) 布尔掩码。
        cross_mask: None 或 (T,S) 布尔掩码。

    Returns:
        返回形状为 `(target_length, model_dim)` 的更新后解码器状态。
    """
    pass
