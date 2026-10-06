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
    x = embedding_weight[token_ids,:]
    for params in blocks:
        def rms(x, scale):
            return scale * x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        def attn(x):
            l, d = x.shape
            n = num_heads
            h = d // n
            def rope(q, k):
                positions = np.arange(l)
                frequencies = 10000 ** (-np.arange(0, h, 2) / h)
                angles = positions[:, None] * frequencies[None, :]
                cosine = np.cos(angles)[None]
                sine = np.sin(angles)[None]
                def rotate(x):
                    ret = np.empty_like(x)
                    x2i = x[..., ::2]
                    x2i1 = x[..., 1::2]
                    ret[..., ::2] = x2i * cosine - x2i1 * sine
                    ret[..., 1::2] = x2i * sine + x2i1 * cosine
                    return ret
                return rotate(q), rotate(k)
            q = (x @ params['w_q']).reshape((l, n, h)).transpose(1, 0, 2)
            k = (x @ params['w_k']).reshape((l, n, h)).transpose(1, 0, 2)
            v = (x @ params['w_v']).reshape((l, n, h)).transpose(1, 0, 2)
            q, k = rope(q, k)
            scores = np.where(np.tril(np.ones((l, l))), (q @ k.transpose(0, 2, 1)) / np.sqrt(h), -np.inf)
            def softmax(x):
                x = x - x.max(axis=-1, keepdims=True)
                ex = np.exp(x)
                return ex / ex.sum(axis=-1, keepdims=True)
            return (softmax(scores) @ v).transpose(1, 0, 2).reshape(x.shape) @ params['w_o']
        def ffn(x):
            up = x @ params['w_up']
            gate = x @ params['w_gate']
            return (up * gate / (1 + np.exp(-gate))) @ params['w_down']
        y = x + attn(rms(x, params['norm1_scale']))
        z = y + ffn(rms(y, params['norm2_scale']))
        x = z
    return rms(x, final_norm_scale) @ embedding_weight.T