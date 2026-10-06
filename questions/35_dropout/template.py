import numpy as np


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
    pass
