# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：把当前分块产生的 K/V 追加到历史缓存，只为新 Query 计算因果注意力。
# API：np.concatenate(..., axis=1) 沿时间轴追加缓存；np.repeat 扩展 GQA 的 KV 头。

def kv_cache_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    cache_k: np.ndarray | None = None,
    cache_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        x: 当前分块，形状为 (chunk_length, model_dim)。
        w_q: 形状为 (D,D) 的查询投影。
        w_k / w_v: 形状为 (D,num_kv_heads*head_dim) 的 KV 投影。
        w_o: 形状为 (D,D) 的输出投影。
        num_query_heads / num_kv_heads: 查询头数与 KV 头数，前者可被后者整除。
        cache_k / cache_v: None 或形状为 (num_kv_heads,past_length,head_dim) 的历史缓存。

    Returns:
        返回 `(output, new_cache_k, new_cache_v)`；output 为 `(chunk_length,D)`，新缓存长度为 past_length+chunk_length。
    """
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    chunk_length, model_dim = x.shape
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = model_dim // num_query_heads
    # 当前分块生成 Q 以及需要追加到缓存的新 K/V。
    q = (x @ w_q).reshape(chunk_length, num_query_heads, head_dim).transpose(1, 0, 2)
    # 当前分块投影出新的 Key，并重排为 (kv_head,chunk,head_dim)，准备追加到历史缓存。
    new_k = (x @ w_k).reshape(chunk_length, num_kv_heads, head_dim).transpose(1, 0, 2)
    # 当前分块投影出新的 Value，使用与 Key 相同的缓存布局。
    new_v = (x @ w_v).reshape(chunk_length, num_kv_heads, head_dim).transpose(1, 0, 2)
    # 没有历史缓存时，当前分块的 K/V 就是完整缓存，历史长度为 0。
    if cache_k is None:
        # 首个分块没有历史缓存，past 从零开始。
        all_k, all_v, past = new_k, new_v, 0
    # 以上专用分支均不匹配时，执行这里约定的剩余算法分支。
    else:
        # axis=1 是时间轴，历史 K/V 必须放在新分块之前。
        past = cache_k.shape[1]
        # 把历史 Key 与当前分块的新 Key 沿时间轴连接。 `np.concatenate(..., axis=1)` 沿时间轴拼接且不新增维度。
        all_k = np.concatenate((cache_k, new_k), axis=1)
        # 把历史 Value 与当前分块的新 Value 沿时间轴连接。 `np.concatenate(..., axis=1)` 沿时间轴拼接且不新增维度。
        all_v = np.concatenate((cache_v, new_v), axis=1)
    # GQA 中重复较少的 KV 头，使头数与 Query 对齐。
    repeat = num_query_heads // num_kv_heads
    # `np.repeat(..., axis=0)` 把每个缓存 Key 头连续复制，使头数与 Query 对齐。
    expanded_k = np.repeat(all_k, repeat, axis=0)
    # 对缓存 Value 做相同的头复制，保证每个 Query 头读取匹配的 Value。
    expanded_v = np.repeat(all_v, repeat, axis=0)
    # 新 Query 与完整缓存 Key 做批量点积，并除以 `sqrt(head_dim)` 控制分数尺度。
    scores = q @ expanded_k.transpose(0, 2, 1) / np.sqrt(head_dim)
    # 第 i 个新 query 可看全部 past，以及当前分块中不晚于 i 的位置。
    mask = np.arange(all_k.shape[1])[None, :] <= past + np.arange(chunk_length)[:, None]
    # `mask[None]` 把同一因果掩码广播到全部头；`np.where` 令不可见 Key 的分数为负无穷。
    scores = np.where(mask[None], scores, -np.inf)
    # 稳定 Softmax 后汇总扩展后的 V，最终拼头并输出投影。
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
    weights /= weights.sum(axis=-1, keepdims=True)
    # 注意力权重汇总扩展后的 V，随后拼接 Query 头并通过 w_o 投影回模型维。
    output = (weights @ expanded_v).transpose(1, 0, 2).reshape(chunk_length, model_dim) @ w_o
    # 返回当前分块输出以及包含历史和新 token 的完整 K/V 缓存。
    return output, all_k, all_v
