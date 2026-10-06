# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：训练时随机屏蔽并按保留概率缩放激活，推理时原样返回。
# API：random(shape) 生成 [0,1) 均匀样本；与 probability 比较会得到同形布尔 mask。

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
    # 推理模式或零丢弃率无需随机采样，直接走恒等路径并返回全 True 掩码。
    if not training or probability == 0.0:
        # 推理或零丢弃率下原样复制，并报告全保留掩码。
        return x.copy(), np.ones(x.shape, dtype=bool)
    # 每个位置独立采样；随机数不小于丢弃率时保留。
    mask = np.random.default_rng(seed).random(x.shape) >= probability
    # inverted dropout 除以保留概率，使训练输出的期望等于原输入。
    return x * mask / (1.0 - probability), mask
