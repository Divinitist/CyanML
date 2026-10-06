# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：串联解码器自注意力、对编码器输出的交叉注意力和前馈子层。
# 语法：prefix 拼接字典键，让同一 attend 函数复用 self/cross 两套投影参数。

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
    # 定义带参数前缀的 LayerNorm，使三个子层复用同一实现。
    def norm(value, prefix):
        # prefix 选择对应子层的 LayerNorm 缩放与偏置参数。
        mean = value.mean(axis=-1, keepdims=True)
        # 计算每个 token 最后一维的方差，并保留单例维度供广播。
        var = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
        # 用 prefix 选择当前子层的 scale/bias，对每个 token 的最后一维执行 LayerNorm 仿射变换。
        return (value - mean) / np.sqrt(var + 1e-5) * params[prefix + "_scale"] + params[prefix + "_bias"]

    # 定义可同时服务自注意力和交叉注意力的局部函数，参数前缀决定所用权重。
    def attend(query, context, prefix, mask):
        # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
        q_len, dim = query.shape
        # 从数组形状读取 `k_len`，供后续维度计算使用。
        k_len = context.shape[0]
        # 用模型维度除以头数，得到每个注意力头负责的特征数。
        head_dim = dim // num_heads
        # self 前缀让三者同源；cross 前缀让 Q 来自解码器、K/V 来自编码器。
        q = (query @ params[prefix + "_q"]).reshape(q_len, num_heads, head_dim).transpose(1, 0, 2)
        # 线性投影得到 Key，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
        k = (context @ params[prefix + "_k"]).reshape(k_len, num_heads, head_dim).transpose(1, 0, 2)
        # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
        v = (context @ params[prefix + "_v"]).reshape(k_len, num_heads, head_dim).transpose(1, 0, 2)
        # 得分最后两维是 (query_length, context_length)，mask 沿头维广播。
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
        # 只有调用方提供可见性 mask 时才屏蔽注意力位置；None 表示全可见。
        if mask is not None:
            # `mask[None]` 插入头轴并广播；`np.where` 把不可见位置设为负无穷。
            scores = np.where(mask[None], scores, -np.inf)
        # 数值稳定的 Softmax 将每个 query 的 key 分数归一化。
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
        weights /= weights.sum(axis=-1, keepdims=True)
        # 权重乘 V 后拼接全部头，再用所选前缀的输出矩阵映射回 dim。
        return (weights @ v).transpose(1, 0, 2).reshape(q_len, dim) @ params[prefix + "_o"]

    # 三个 Pre-Norm 子层依次是因果自注意力、交叉注意力、前馈网络。
    y_input = norm(decoder_x, "ln1")
    # 把因果自注意力输出加回原解码器状态，形成第一条残差 y。
    y = decoder_x + attend(y_input, y_input, "self", self_mask)
    # 用归一化后的 y 查询 encoder_x，并把交叉注意力输出加回形成第二条残差 z。
    z = y + attend(norm(y, "ln2"), encoder_x, "cross", cross_mask)
    # FFN 先升维并做 GELU，降维结果通过第三条残差加回 z。
    hidden = norm(z, "ln3") @ params["ff1"] + params["ff1_bias"]
    # 对 FFN 升维结果应用 GELU 的 tanh 近似，提供非线性。
    hidden = .5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + .044715 * hidden ** 3)))
    # 用 ff2 把隐藏维降回模型维、加偏置，再作为第三条残差加回 z。
    return z + hidden @ params["ff2"] + params["ff2_bias"]
