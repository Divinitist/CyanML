# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：贪心生成后续 token，并比较全量重算与 KV Cache 增量解码两条路径。
# API：np.concatenate 沿时间轴追加缓存；np.stack 把每一步的一维 logits 叠成二维结果。

def autoregressive_generate(
    prompt_tokens: np.ndarray,
    embedding_weight: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    max_new_tokens: int,
    end_token: int,
    use_cache: bool,
) -> tuple[list[int], np.ndarray]:
    """
    Args:
        prompt_tokens: 形状为 (prompt_length,) 的非空整数前缀。
        embedding_weight: 形状为 (vocab_size,model_dim) 的共享嵌入/LM Head。
        params: 含四个 (model_dim,model_dim) 注意力矩阵的字典。
        num_heads: 能整除 model_dim 的查询/KV 头数。
        max_new_tokens: 最多生成的 token 数。
        end_token: 生成后提前停止的合法 token 下标。
        use_cache: 是否启用 KV Cache 增量路径。

    Returns:
        返回 `(tokens, step_logits)`；tokens 是包含原 prompt 和新 token 的列表，step_logits 形状为 `(实际生成步数,vocab_size)`。
    """
    # 转成 Python int 列表，便于每步 append 新 token。
    tokens = [int(token) for token in prompt_tokens]
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = embedding_weight.shape[1] // num_heads

    # 定义可同时服务自注意力和交叉注意力的局部函数，参数前缀决定所用权重。
    def attend(chunk_tokens, cache_k=None, cache_v=None):
        # chunk 可以是完整上下文，也可以是缓存模式下刚生成的单个 token。
        x = embedding_weight[np.asarray(chunk_tokens)]
        # 从数组形状读取 `chunk, dim`，供后续维度计算使用。
        chunk, dim = x.shape
        # 线性投影得到 Query，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
        q = (x @ params["w_q"]).reshape(chunk, num_heads, head_dim).transpose(1, 0, 2)
        # 为当前 chunk 投影 Key，并转为 (head,chunk,head_dim) 以便沿时间轴追加缓存。
        new_k = (x @ params["w_k"]).reshape(chunk, num_heads, head_dim).transpose(1, 0, 2)
        # 为当前 chunk 投影 Value，缓存布局与 Key 保持一致。
        new_v = (x @ params["w_v"]).reshape(chunk, num_heads, head_dim).transpose(1, 0, 2)
        # cache 的 axis=1 是历史时间长度；新 K/V 沿该轴追加。
        past = 0 if cache_k is None else cache_k.shape[1]
        # 把历史 Key 与当前分块的新 Key 沿时间轴连接。 `np.concatenate(..., axis=1)` 沿时间轴拼接且不新增维度。
        all_k = new_k if cache_k is None else np.concatenate((cache_k, new_k), axis=1)
        # 把历史 Value 与当前分块的新 Value 沿时间轴连接。 `np.concatenate(..., axis=1)` 沿时间轴拼接且不新增维度。
        all_v = new_v if cache_v is None else np.concatenate((cache_v, new_v), axis=1)
        # 新 Query 与历史及当前 K 全部点积，但因果 mask 限制可见范围。
        scores = q @ all_k.transpose(0, 2, 1) / np.sqrt(head_dim)
        # 广播比较 Key 位置与每个新 Query 的绝对位置，生成 chunk 对完整缓存的因果掩码。
        mask = np.arange(all_k.shape[1])[None, :] <= past + np.arange(chunk)[:, None]
        # 掩码后执行数值稳定的 Softmax，得到每个头的注意力权重。
        scores = np.where(mask[None], scores, -np.inf)
        # 先对平移后的注意力分数逐元素取指数，得到未归一化注意力权重。 `np.exp` 对数组逐元素取自然指数。
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
        weights /= weights.sum(axis=-1, keepdims=True)
        # 注意力输出加输入残差，再用共享嵌入矩阵投影到词表 logits。
        hidden = x + (weights @ all_v).transpose(1, 0, 2).reshape(chunk, dim) @ params["w_o"]
        # 缓存模式返回新旧 K/V 合并结果，使下一生成步无需重算历史 token。
        return hidden @ embedding_weight.T, all_k, all_v

    # 建立列表保存每个生成步骤的完整词表 logits，结束时再沿时间轴堆叠。
    step_logits = []
    # 同时把两个缓存初始化为 None，表示尚未处理 prompt。
    cache_k = cache_v = None
    # 缓存模式先用完整 prompt 预填充 K/V，之后才能逐 token 增量解码。
    if use_cache:
        # 先一次性预填充 prompt 缓存，之后每步只处理一个新 token。
        logits, cache_k, cache_v = attend(tokens)
    # 逐步扩展序列，最多生成指定数量的新 token。
    for _ in range(max_new_tokens):
        # 无缓存模式必须在每一步对不断增长的完整序列重算注意力。
        if not use_cache:
            # 无缓存路径每一步都用增长后的完整 tokens 重算注意力。
            logits, _, _ = attend(tokens)
        # 贪心选择当前最后位置 logit 最大的 token，并保存该步调试输出。
        current = logits[-1].copy()
        # `append` 保存本步完整词表 logits，最终用 `np.stack` 按生成步拼成二维数组。
        step_logits.append(current)
        # `np.argmax` 取当前 logits 最大的词表下标实现贪心解码；`int` 转成普通 Python 整数。
        next_token = int(np.argmax(current))
        # 把贪心选出的 token 追加到已生成序列，使下一步能把它作为上下文。
        tokens.append(next_token)
        # 命中 EOS 立即停止，不再把 EOS 送入下一轮注意力。
        if next_token == end_token:
            # 收敛或终止条件已经满足，提前退出迭代以避免无效计算。
            break
        # 缓存模式先用完整 prompt 预填充 K/V，之后才能逐 token 增量解码。
        if use_cache:
            # 增量路径只投影新 token，并把产生的 K/V 追加到缓存。
            logits, cache_k, cache_v = attend([next_token], cache_k, cache_v)
    # 无生成步骤时显式返回形状为 (0, vocab_size) 的空 logits。
    return tokens, np.stack(step_logits) if step_logits else np.empty((0, embedding_weight.shape[0]))
