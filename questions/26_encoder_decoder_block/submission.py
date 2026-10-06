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
    def gelu(value):
        return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * value ** 3)))
    def ln(x, id):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return params[f'ln{id}_scale'] * (x - mean) / np.sqrt(var + 1e-5) + params[f'ln{id}_bias']
    def ffn(x):
        return gelu(x @ params['ff1'] + params['ff1_bias']) @ params['ff2'] + params['ff2_bias']
    def attn(enx, dex, mask, mode):
        kl, d = enx.shape # 卧槽，encoder 的结果才是 context，decoder 的结果是 query
        ql = dex.shape[0]
        n = num_heads
        h = d // n
        q = (dex @ params[f'{mode}_q']).reshape((ql, n, h)).transpose(1, 0, 2)
        k = (enx @ params[f'{mode}_k']).reshape((kl, n, h)).transpose(1, 0, 2)
        v = (enx @ params[f'{mode}_v']).reshape((kl, n, h)).transpose(1, 0, 2)
        scores = np.where(mask[None], (q @ k.transpose(0, 2, 1)) / np.sqrt(h), -np.inf)
        def softmax(x):
            x = x - x.max(axis=-1, keepdims=True)
            ex = np.exp(x)
            return ex / ex.sum(axis=-1, keepdims=True)
        attn = (softmax(scores) @ v).transpose(1, 0, 2).reshape((ql, d))
        return attn @ params[f'{mode}_o']
    lnx = ln(decoder_x, 1)
    y = decoder_x + attn(lnx, lnx, self_mask, 'self')
    lny = ln(y, 2)
    z = y + attn(encoder_x, lny, cross_mask, 'cross')
    return z + ffn(ln(z, 3))
