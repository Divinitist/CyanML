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
    pass
