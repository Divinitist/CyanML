import numpy as np


def decoder_only_transformer(
    token_ids: np.ndarray,
    embedding_weight: np.ndarray,
    blocks: list[dict[str, np.ndarray]],
    final_norm_scale: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Args:
        token_ids: 形状为 (sequence_length,) 的整数 token 序列。
        embedding_weight: 形状为 (vocab_size,model_dim) 的输入/输出共享权重。
        blocks: 一个或多个解码器块参数字典组成的列表。
        final_norm_scale: 形状为 (model_dim,) 的最终 RMSNorm 缩放。
        num_heads: 能整除 model_dim 且头维为偶数的头数。

    Returns:
        返回形状为 `(sequence_length,vocab_size)` 的 next-token logits。
    """
    pass
