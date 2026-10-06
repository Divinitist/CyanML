import numpy as np


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
    # 1. 排序，过滤出 topk 对应的 id 序列
    # 2. 把没包含在内的 id 的对应值全部置为 -inf
    # 3. 计算概率
    # 4. 排序，堆积概率直到超过 topp，记下对应的 id 序列
    # 5. 把没在 id 序列内的位置置为 -inf
    # 6. 重新计算概率
    # 7. 把概率给 rng 抽卡
    logits /= temperature
    def get_probs(logits):
        x = logits
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)
    if top_k:
        x = sorted(enumerate(logits), key=lambda x: -x[1])[top_k:]
        if x:
            ids = np.array([i for i, _ in x])
            logits[ids] = -np.inf
    probs = get_probs(logits)
    x = sorted(enumerate(probs), key=lambda x:-x[1])
    sum = 0
    ids = []
    for i in range(len(x)-1, -1, -1):
        if sum + x[i][1] >= 1-top_p:
            break
        sum += x[i][1]
        ids.append(int(x[i][0]))
    if ids:
        logits[np.array(ids)] = -np.inf
    probs = get_probs(logits)
    return np.random.default_rng(seed).choice(np.arange(logits.shape[0]), p=probs), probs
