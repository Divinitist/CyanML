# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：用 query 序列产生 Q、用 context 序列产生 K/V，完成跨序列信息读取。
# API：transpose(0, 2, 1) 只交换每个头的序列轴与特征轴，供 Q @ K^T 使用。

def cross_attention(
    query: np.ndarray,
    context: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            query: 形状为 (query_length, model_dim)。
            context: 形状为 (context_length, model_dim)。
            w_q / w_k / w_v / w_o: 形状均为 (model_dim, model_dim)。
            num_heads: 能整除模型维度的头数。
            mask: None 或形状为 (Q,K) 的布尔掩码。

        Returns:
            返回形状为 `(query_length, model_dim)` 的 float64 数组。
        """
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    query_length, model_dim = query.shape
    # 读取编码器上下文的 token 数，决定 K/V 的序列轴长度。
    context_length = context.shape[0]
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = model_dim // num_heads
    # Q 来自 query，K/V 来自 context，这是交叉注意力与自注意力的关键区别。
    q = (query @ w_q).reshape(query_length, num_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Key，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    k = (context @ w_k).reshape(context_length, num_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    v = (context @ w_v).reshape(context_length, num_heads, head_dim).transpose(1, 0, 2)
    # 得分矩阵最后两维为 (query_length, context_length)。
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    # 只有调用方提供可见性 mask 时才屏蔽注意力位置；None 表示全可见。
    if mask is not None:
        # 同一张 (Q, K) 掩码沿头维广播。
        scores = np.where(mask[None, :, :], scores, -np.inf)
    # 行最大值平移防止 exp 上溢，再沿 context 位置归一化。
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
    weights /= weights.sum(axis=-1, keepdims=True)
    # 每个 query 汇总 context 的 V，随后拼头并输出投影。
    joined = (weights @ v).transpose(1, 0, 2).reshape(query_length, model_dim)
    # 输出投影把跨注意力各头拼接结果映射回模型维。
    return joined @ w_o
