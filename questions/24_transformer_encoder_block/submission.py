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
    # 这个好，真正的大战
    # pre 和 post 的区别在于在 attn 和 res-con 之前/之后的过程
    def ln1(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return params['ln1_scale'] * (x - mean) / np.sqrt(var + 1e-5) + params['ln1_bias']
    def ln2(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return params['ln2_scale'] * (x - mean) / np.sqrt(var + 1e-5) + params['ln2_bias']
    def attn(x):
        L, D = x.shape
        N = num_heads
        H = D // N
        q = (x @ params['w_q']).reshape((L, N, H)).transpose(1, 0, 2)
        k = (x @ params['w_k']).reshape((L, N, H)).transpose(1, 0, 2)
        v = (x @ params['w_v']).reshape((L, N, H)).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(H)
        if mask is not None:
            scores = np.where(mask[None,:,:], scores, -np.inf)
        def softmax(x):
            x = x - x.max(axis=-1,keepdims=True)
            ex = np.exp(x)
            return ex / ex.sum(axis=-1,keepdims=True)
        weights = softmax(scores)
        attn = (weights @ v).transpose(1, 0, 2).reshape((L, D))
        return attn @ params['w_o']
    # def ffn(x):
    #     return np.tanh(x @ params['w1'] + params['b1']) @ params['w2'] + params['b2']
    def ffn(value):
        hidden = value @ params["w1"] + params["b1"]
        hidden = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
        return hidden @ params["w2"] + params["b2"]
    if pre_norm:
        y = x + attn(ln1(x))
        z = y + ffn(ln2(y))
    else:
        y = ln1(x + attn(x))
        z = ln2(y + ffn(y))
    return z
