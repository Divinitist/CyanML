# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：从 token 嵌入开始顺序执行多个因果解码器块，最终输出共享 LM Head logits。
# API：np.tril 构造因果下三角；内部 rotate 用偶/奇切片原地组装 RoPE 结果。

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
    # 定义 RMSNorm 辅助函数，供每个解码器块和最终输出层复用。
    def rms(value, scale):
        # 每个 token 独立按最后一维均方根归一化，不做均值中心化。
        return value / np.sqrt(np.mean(value ** 2, axis=-1, keepdims=True) + 1e-6) * scale

    # 高级索引查出整个序列的 token 嵌入，copy 避免后续残差修改权重表视图。
    hidden = embedding_weight[token_ids].copy()
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    length, dim = hidden.shape
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = dim // num_heads
    # 所有块共享相同位置和 RoPE 三角函数，提前计算可避免重复工作。
    positions = np.arange(length)
    # `np.arange(0, head_dim, 2)` 为每对通道生成频率下标，负幂让高维通道旋转更慢。
    frequencies = 10000.0 ** (-np.arange(0, head_dim, 2) / head_dim)
    # 通过位置列向量与频率行向量广播，生成每个位置、每个维度对的旋转角。
    angles = positions[:, None] * frequencies[None, :]
    # 一次计算并解包 RoPE 所需的余弦表与正弦表，两者形状相同。
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    # 按模型深度顺序执行每个解码器块，前一块输出作为后一块输入。
    for params in blocks:
        # 每个解码器块采用 Pre-RMSNorm，并由归一化状态产生 Q/K/V。
        normalized = rms(hidden, params["norm1_scale"])
        # Query 先保持 (sequence,head,head_dim)，让 RoPE 的位置轴可直接广播。
        q = (normalized @ params["w_q"]).reshape(length, num_heads, head_dim)
        # Key 同样先保持 sequence-first，旋转完成后再与 Query 一起转为 head-first。
        k = (normalized @ params["w_k"]).reshape(length, num_heads, head_dim)
        # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
        v = (normalized @ params["w_v"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
        # 定义 RoPE 二维旋转辅助函数，对偶数/奇数通道成对处理。
        def rotate(value):
            # 将相邻偶/奇通道作为二维向量做位置相关旋转。
            result = np.empty_like(value)
            # 用步长为 2 的切片同时解包偶数和奇数通道，组成二维旋转对。
            even, odd = value[..., 0::2], value[..., 1::2]
            # 按二维旋转公式写入这一半通道；省略号保留序列轴和头轴。
            result[..., 0::2] = even * cosine - odd * sine
            # 按二维旋转公式写入这一半通道；省略号保留序列轴和头轴。
            result[..., 1::2] = even * sine + odd * cosine
            # `transpose(1,0,2)` 把旋转结果改为 (head,sequence,head_dim)，供批量点积注意力使用。
            return result.transpose(1, 0, 2)
        # 只旋转 Q/K；V 保持原值，注意力分数按头维平方根缩放。
        q, k = rotate(q), rotate(k)
        # 每个头计算 `Q @ K^T` 的位置相似度，并除以 `sqrt(head_dim)` 防止大头维令 Softmax 过度尖锐。
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
        # 下三角掩码阻止 token 读取未来位置。
        scores = np.where(np.tril(np.ones((length, length), dtype=bool))[None], scores, -np.inf)
        # 稳定 Softmax 加权 V，拼头后经过 w_o 并加入注意力残差。
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
        weights /= weights.sum(axis=-1, keepdims=True)
        # 汇总 V、拼接各头并做输出投影，再加回旧 hidden 形成注意力残差。
        hidden = hidden + (weights @ v).transpose(1, 0, 2).reshape(length, dim) @ params["w_o"]
        # 第二次 Pre-Norm 后执行 SwiGLU：SiLU(gate) 与 up 逐元素相乘。
        normalized = rms(hidden, params["norm2_scale"])
        # 门投影产生 SwiGLU 的 gate 分支，后面会逐元素应用 SiLU。
        gate = normalized @ params["w_gate"]
        # 内容投影产生与 gate 同形的 up 分支，供门控乘法使用。
        up = normalized @ params["w_up"]
        # 计算 `SiLU(gate)*up`，经 w_down 降回模型维后加回 hidden，形成 FFN 残差。
        hidden = hidden + (gate / (1 + np.exp(-gate)) * up) @ params["w_down"]
    # 最终 RMSNorm 后复用 embedding_weight.T 作为 tied LM Head。
    return rms(hidden, final_norm_scale) @ embedding_weight.T
