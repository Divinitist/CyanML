# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：依次应用温度、top-k、top-p 过滤，并从最终词表分布采样一个 token。
# API：np.cumsum 得到累计概率；searchsorted 找到首次达到 top-p 的有序位置。

def sample_next_token(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> tuple[int, np.ndarray]:
    """
    Args:
        logits: 形状为 (vocab_size,) 的 float64 未归一化分数。
        temperature: 大于 0 的温度。
        top_k: 0 或不超过词表大小的候选数。
        top_p: 位于 (0,1] 的核采样阈值。
        seed: 单次采样的整数随机种子。

    Returns:
        返回 `(token_id, probabilities)`；token_id 为 Python int，probabilities 形状为 `(vocab_size,)`、总和为 1。
    """
    # 温度小于 1 会放大分数差异，大于 1 会使分布更平坦。
    scaled = logits / temperature
    # 建立词表布尔保留掩码，后续 top-k 过滤通过它选择候选 token。
    keep = np.ones(logits.size, dtype=bool)
    # 仅当 top_k 真正缩小词表时执行过滤；0 或不小于词表大小表示禁用 top-k。
    if top_k > 0 and top_k < logits.size:
        # stable 降序排序在分数相同时优先保留较小 token 下标。
        order = np.argsort(-scaled, kind="stable")
        # 先清空候选集合，再只把 top-k 下标对应位置标记为保留。
        keep[:] = False
        # 先清空候选集合，再只把 top-k 下标对应位置标记为保留。
        keep[order[:top_k]] = True
    # 丢弃项设为负无穷，经 Softmax 后概率正好为零。
    filtered = np.where(keep, scaled, -np.inf)
    # 减去最大有限分数，避免 exp 上溢。
    probabilities = np.exp(filtered - np.max(filtered))
    # 除以概率总和重新归一化，使过滤后保留项之和恢复为 1。
    probabilities /= probabilities.sum()
    # top_p 小于 1 时启用核采样；等于 1 表示保留当前分布的全部非零候选。
    if top_p < 1.0:
        # 按概率从大到小累加，保留累计概率首次达到 top_p 的最小前缀。
        order = np.argsort(-probabilities, kind="stable")
        # 按概率降序计算前缀累计和，用于确定 top-p 核边界。 `np.cumsum` 按当前顺序产生逐项累计和。
        cumulative = np.cumsum(probabilities[order])
        # 找到累计概率首次达到 top-p 的位置，并加一换成保留元素数量。 `searchsorted(..., side='left')` 返回可保持有序的最左插入位置；`int(...)` 把 NumPy 整数转换为普通 Python 整数。
        count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        # 创建与词表同长的核采样布尔掩码。 `np.zeros_like` 创建与参照数组同形、同 dtype 的全零数组。
        nucleus = np.zeros_like(keep)
        # 把累计概率阈值内的有序前缀标成核采样候选。
        nucleus[order[:count]] = True
        # 核外概率清零后必须再次归一化。
        probabilities = np.where(nucleus, probabilities, 0.0)
        # 除以概率总和重新归一化，使过滤后保留项之和恢复为 1。
        probabilities /= probabilities.sum()
    # choice 的 p 参数按最终分布采样；固定 seed 保证结果可复现。
    token = int(np.random.default_rng(seed).choice(logits.size, p=probabilities))
    # 返回采样 token 的 Python 整数下标和过滤、重归一化后的完整词表概率。
    return token, probabilities
