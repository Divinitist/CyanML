# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：组合 LayerNorm、多头自注意力、GELU FFN 和残差，支持 Pre/Post-Norm。
# API：字典按参数名组织权重；局部函数封装会重复调用但只服务本块的计算步骤。

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
    # 定义逐 token 的 LayerNorm，注意力和 FFN 子层会重复调用。
    def layer_norm(value, scale, bias):
        # LayerNorm 对每个 token 的最后一维独立标准化。
        mean = value.mean(axis=-1, keepdims=True)
        # 计算归一化轴上的总体方差；`keepdims=True` 保留轴以便广播。
        variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
        # 标准化后按特征乘 scale、加 bias；一维仿射参数会广播到每个 token。
        return (value - mean) / np.sqrt(variance + 1e-5) * scale + bias

    # 定义本块的多头自注意力子层，闭包直接读取参数字典和 mask。
    def attention(value):
        # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
        length, model_dim = value.shape
        # 用模型维度除以头数，得到每个注意力头负责的特征数。
        head_dim = model_dim // num_heads
        # Q/K/V 拆成 (head, sequence, head_dim)，并行执行各注意力头。
        q = (value @ params["w_q"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
        # 线性投影得到 Key，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
        k = (value @ params["w_k"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
        # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
        v = (value @ params["w_v"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
        # 缩放点积分数过大时会让 Softmax 饱和，因此除以 sqrt(head_dim)。
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
        # 只有调用方提供可见性 mask 时才屏蔽注意力位置；None 表示全可见。
        if mask is not None:
            # `np.where` 保留可见分数、把不可见位置设为负无穷，使其 Softmax 权重为 0。
            scores = np.where(mask[None], scores, -np.inf)
        # 减行最大值后计算稳定 Softmax。
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
        weights /= weights.sum(axis=-1, keepdims=True)
        # 对 V 加权、拼接各头，最后通过输出矩阵混合头信息。
        return (weights @ v).transpose(1, 0, 2).reshape(length, model_dim) @ params["w_o"]

    # 定义逐 token 的两层前馈子层，把模型维升维后再降回原维度。
    def ffn(value):
        # 两层逐 token 前馈网络，中间使用 GELU 非线性。
        hidden = value @ params["w1"] + params["b1"]
        # 使用 tanh 近似式对升维结果逐元素应用 GELU，引入平滑非线性。
        hidden = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
        # 第二个线性层把 FFN 隐藏维降回 model_dim，并加输出偏置。
        return hidden @ params["w2"] + params["b2"]

    # Pre-Norm 分支在进入每个子层前归一化，再把子层输出加回残差。
    if pre_norm:
        # Pre-Norm 在每个子层前归一化，子层输出再接残差。
        y = x + attention(layer_norm(x, params["ln1_scale"], params["ln1_bias"]))
        # 对第一条残差 y 做第二次 Pre-Norm，计算 FFN 后作为第二条残差加回。
        return y + ffn(layer_norm(y, params["ln2_scale"], params["ln2_bias"]))
    # Post-Norm 先做子层和残差，再对相加结果归一化。
    y = layer_norm(x + attention(x), params["ln1_scale"], params["ln1_bias"])
    # Post-Norm 路径先把 FFN 输出加回 y，再对第二条残差整体归一化。
    return layer_norm(y + ffn(y), params["ln2_scale"], params["ln2_bias"])
