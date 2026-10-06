# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：让若干 Query 头共享一个 KV 头，在 MHA 与 MQA 之间折中容量和缓存开销。
# API：np.repeat(..., axis=0) 沿头维复制 KV，使其数量与 Query 头对齐。

def grouped_query_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (L,D) 的 float64 输入。
            w_q: 形状为 (D,D) 的查询投影。
            w_k / w_v: 形状为 (D, num_kv_heads*head_dim)。
            w_o: 形状为 (D,D)。
            num_query_heads / num_kv_heads: 查询头数和 KV 头数。
            mask: None 或 (L,L) 布尔掩码。

        Returns:
            返回形状为 `(L,D)` 的注意力输出。
        """
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    length, model_dim = x.shape
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = model_dim // num_query_heads
    # Q 使用较多的头，K/V 只投影为较少的 num_kv_heads 组。
    q = (x @ w_q).reshape(length, num_query_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Key，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    k = (x @ w_k).reshape(length, num_kv_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    v = (x @ w_v).reshape(length, num_kv_heads, head_dim).transpose(1, 0, 2)
    # 每个 KV 头连续复制 repeat 次，使其与对应的一组 Query 头对齐。
    repeat = num_query_heads // num_kv_heads
    # `np.repeat(..., axis=0)` 把每个 KV 头连续复制 `repeat` 次，使 K 的头数与 Query 对齐。
    k = np.repeat(k, repeat, axis=0)
    # 对 Value 做与 Key 完全相同的头复制，保证每个 Query 头读取匹配的 KV 组。
    v = np.repeat(v, repeat, axis=0)
    # 对齐头数后即可按头执行标准缩放点积注意力。
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    # 只有调用方提供可见性 mask 时才屏蔽注意力位置；None 表示全可见。
    if mask is not None:
        # 负无穷掩码确保不可见位置的 Softmax 权重为零。
        scores = np.where(mask[None, :, :], scores, -np.inf)
    # 先对平移后的注意力分数逐元素取指数，得到未归一化注意力权重。 `np.exp` 对数组逐元素取自然指数。
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
    weights /= weights.sum(axis=-1, keepdims=True)
    # 合并各 Query 头后通过 w_o 混合头间信息。
    context = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim)
    # 输出投影负责混合各 Query 头，返回 GQA 的最终结果。
    return context @ w_o
