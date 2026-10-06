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
    # 整体复习一遍。prenorm，先norm再映射为qkv，再rope qk，再下三角掩码算注意力，残差连接之后再norm，再ffn之后再残差连接
    def rms(x, scale):
        var = np.mean(x ** 2, axis=-1, keepdims=True)
        return scale * x / np.sqrt(var + 1e-6)
    def attn(x):
        l, d = x.shape
        n = num_heads
        h = d // n
        q = (x @ params['w_q']).reshape((l, n, h)).transpose(1, 0, 2)
        k = (x @ params['w_k']).reshape((l, n, h)).transpose(1, 0, 2)
        v = (x @ params['w_v']).reshape((l, n, h)).transpose(1, 0, 2)
        def rope(q, k):
            angles = np.arange(l)[:,None] * (10000 ** (-np.arange(0, h, 2) / h))[None,:]
            cosine = np.cos(angles)[None] # [None] 是 [None,:,:] 的简写
            sine = np.sin(angles)[None]
            def rotate(x):
                ret = x.copy()
                x2i = x[..., ::2]
                x2i1 = x[..., 1::2]
                ret[..., ::2] = x2i * cosine - x2i1 * sine
                ret[..., 1::2] = x2i * sine + x2i1 * cosine
                return ret
            return rotate(q), rotate(k)
        q, k = rope(q, k)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(h)
        scores = np.where(np.tril(np.ones((l, l)))[None], scores, -np.inf)
        def softmax(x):
            x = x - x.max(axis=-1, keepdims=True)
            ex = np.exp(x)
            return ex / ex.sum(axis=-1, keepdims=True)
        weights = softmax(scores)
        attn = (weights @ v).transpose(1, 0, 2).reshape(x.shape)
        return attn @ params['w_o']
    def ffn(x):
        up = x @ params['w_up']
        gate = x @ params['w_gate']
        return (gate / (1 + np.exp(-gate)) * up) @ params['w_down']
    y = x + attn(rms(x, params['norm1_scale']))
    z = y + ffn(rms(y, params['norm2_scale']))
    return z
