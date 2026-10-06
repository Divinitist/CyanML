import numpy as np
from random import random

def dropout(
    x: np.ndarray,
    probability: float,
    training: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        x: 任意形状的 float64 输入数组。
        probability: 满足 0<=p<1 的丢弃概率。
        training: True 表示训练模式，False 表示推理模式。
        seed: 生成掩码的整数随机种子。

    Returns:
        返回 `(output, keep_mask)`；output 与 x 同形，keep_mask 是同形布尔数组。
    """
    # dropout 原理：每个位置独立判断是否丢弃，如果丢弃则为 0。最后除以 (1-p) 是为了保证期望不变。
    if training and probability > 0:
        mask = (np.random.default_rng(seed).random(x.shape) >= probability)
        return x * mask / (1-probability), mask
    else:
        return x, np.ones(x.shape)
