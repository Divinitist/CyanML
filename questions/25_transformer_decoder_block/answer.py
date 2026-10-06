# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：组合 RMSNorm、RoPE 因果自注意力、SwiGLU 与残差，形成解码器块。
# API：np.tril 取下三角生成因果 mask；np.ones(..., dtype=bool) 先创建全可见布尔矩阵。

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
    # 定义 RMSNorm 辅助函数，只按均方根缩放而不减均值。
    def rms_norm(value, scale):
        # RMSNorm 只按均方根缩放，不减去均值。
        return value / np.sqrt(np.mean(value ** 2, axis=-1, keepdims=True) + 1e-6) * scale

    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    length, model_dim = x.shape
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = model_dim // num_heads
    # 第一处 Pre-Norm 的结果同时用于生成 Q/K/V。
    normalized = rms_norm(x, params["norm1_scale"])
    # 线性投影并 `reshape` 为 (sequence,head,head_dim)；Query 会在 RoPE 辅助函数末尾再转成 head-first。
    q = (normalized @ params["w_q"]).reshape(length, num_heads, head_dim)
    # Key 暂存为 (sequence,head,head_dim)，以便 RoPE 的位置轴与 position_ids 直接对齐。
    k = (normalized @ params["w_k"]).reshape(length, num_heads, head_dim)
    # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    v = (normalized @ params["w_v"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    # RoPE 为每对偶/奇通道构造随维度递减的旋转频率。
    frequencies = 10000.0 ** (-np.arange(0, head_dim, 2) / head_dim)
    # 通过位置列向量与频率行向量广播，生成每个位置、每个维度对的旋转角。
    angles = position_ids[:, None] * frequencies[None, :]
    # 一次计算并解包 RoPE 所需的余弦表与正弦表，两者形状相同。
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    # 定义 RoPE 二维旋转辅助函数，对偶数/奇数通道成对处理。
    def rotate(value):
        # 把相邻偶/奇通道当成二维向量旋转，再转成 head-first 布局。
        result = np.empty_like(value)
        # 用步长为 2 的切片同时解包偶数和奇数通道，组成二维旋转对。
        even, odd = value[..., 0::2], value[..., 1::2]
        # 按二维旋转公式写入这一半通道；省略号保留序列轴和头轴。
        result[..., 0::2] = even * cosine - odd * sine
        # 按二维旋转公式写入这一半通道；省略号保留序列轴和头轴。
        result[..., 1::2] = even * sine + odd * cosine
        # `transpose(1,0,2)` 把旋转结果从 (sequence,head,dim) 改为注意力计算需要的 (head,sequence,dim)。
        return result.transpose(1, 0, 2)
    # 分别旋转 Query 和 Key；Value 不参与 RoPE。
    q, k = rotate(q), rotate(k)
    # 下三角布尔矩阵是因果掩码，禁止当前位置读取未来 token。
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    # `np.tril(np.ones(..., bool))` 生成下三角因果掩码；`np.where` 把未来位置分数设为负无穷。
    scores = np.where(np.tril(np.ones((length, length), dtype=bool))[None], scores, -np.inf)
    # 稳定 Softmax 后加权 V，并拼接所有注意力头。
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
    weights /= weights.sum(axis=-1, keepdims=True)
    # 权重乘 V 得到每头上下文，`transpose+reshape` 拼头，最后 `@ w_o` 做输出投影。
    attention = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim) @ params["w_o"]
    # 自注意力输出先与原输入组成第一条残差连接。
    hidden = x + attention
    # 用标准差或均方根缩放输入，得到无量纲的归一化结果。
    normalized = rms_norm(hidden, params["norm2_scale"])
    # SwiGLU 用 SiLU(gate) 逐元素调制内容分支 up。
    gate = normalized @ params["w_gate"]
    # 内容投影生成与 gate 同形的 up 分支，稍后与 SiLU(gate) 逐元素相乘。
    up = normalized @ params["w_up"]
    # `gate/(1+exp(-gate))` 是 SiLU；与 up 相乘后经 w_down 降回 model_dim，形成 SwiGLU 输出。
    ffn = (gate / (1.0 + np.exp(-gate)) * up) @ params["w_down"]
    # 把 SwiGLU 输出作为第二条残差加回注意力隐藏状态。
    return hidden + ffn
