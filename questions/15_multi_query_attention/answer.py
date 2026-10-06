# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：让多个 Query 头共享一组 K/V，在减少 KV 参数和缓存的同时完成注意力。
# API：np.einsum 的字母描述各轴，"hqd,kd->hqk" 表示 Q 与共享 K 的批量点积。

def multi_query_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_query_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (L,D) 的 float64 输入。
            w_q: 形状为 (D,D)。
            w_k / w_v: 形状为 (D, head_dim)。
            w_o: 形状为 (D,D)。
            num_query_heads: 查询头数，head_dim=D/H。
            mask: None 或形状为 (L,L) 的布尔掩码。

        Returns:
            返回形状为 `(L,D)` 的 float64 数组。
        """
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    length, model_dim = x.shape
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = model_dim // num_query_heads
    # Q 拆成多个头，而 K/V 各只保留一组供所有查询头共享。
    q = (x @ w_q).reshape(length, num_query_heads, head_dim).transpose(1, 0, 2)
    # 只生成一组共享 Key，形状为 (sequence, head_dim)，不拆 Query 头。
    k = x @ w_k
    # 只生成一组共享 Value，所有 Query 头都通过 einsum 读取这份内容。
    v = x @ w_v
    # einsum 对每个头的 Q 与同一份 K 做点积，输出轴为 (head, query, key)。
    scores = np.einsum("hqd,kd->hqk", q, k) / np.sqrt(head_dim)
    # 只有调用方提供可见性 mask 时才屏蔽注意力位置；None 表示全可见。
    if mask is not None:
        # 在头维前插入轴，让同一张掩码广播到全部查询头。
        scores = np.where(mask[None, :, :], scores, -np.inf)
    # 稳定 Softmax 将每个 query 对所有 key 的分数归一化。
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
    weights /= weights.sum(axis=-1, keepdims=True)
    # 用共享 V 求加权和，再拼接所有查询头并做输出投影。
    context = np.einsum("hqk,kd->hqd", weights, v).transpose(1, 0, 2).reshape(length, model_dim)
    # 用 `w_o` 混合各 Query 头，返回与输入相同的模型维度。
    return context @ w_o
