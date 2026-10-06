# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：用统一入口实现 BatchNorm、LayerNorm 和 RMSNorm 的核心归一化计算。
# API：axis 指定统计维；keepdims=True 保持二维形状，使均值和方差可直接广播回输入。

def normalization(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mode: str,
    epsilon: float,
) -> np.ndarray:
    """
    Args:
        x: Shape (batch_size, feature_size), dtype float64.
        gamma: Shape (feature_size,), dtype float64.
        beta: Shape (feature_size,), dtype float64.
        mode: One of "batch", "layer", or "rms".
        epsilon: Positive numerical stabilizer.

    Returns:
        Normalized values with the same shape as x.

    BatchNorm reduces axis 0 so each feature has batch statistics. LayerNorm
    and RMSNorm reduce axis 1 while retaining a column for broadcasting.
    RMSNorm deliberately keeps the original mean and divides only by RMS.
    """
    # BatchNorm 分支沿 batch 轴统计，让每个特征独立归一化。
    if mode == "batch":
        # axis=0 按批统计，让每个特征拥有独立均值和方差。
        mean = np.mean(x, axis=0, keepdims=True)
        # 计算归一化轴上的总体方差；`keepdims=True` 保留轴以便广播。 `np.mean` 沿指定轴求算术平均。
        variance = np.mean((x - mean) ** 2, axis=0, keepdims=True)
        # 用标准差或均方根缩放输入，得到无量纲的归一化结果。 `np.sqrt` 对数组逐元素开平方。
        normalized = (x - mean) / np.sqrt(variance + epsilon)
    # LayerNorm 分支沿特征轴统计，让每个样本独立归一化。
    elif mode == "layer":
        # axis=1 按样本统计；保留列维度后可直接广播回该行。
        mean = np.mean(x, axis=1, keepdims=True)
        # 计算归一化轴上的总体方差；`keepdims=True` 保留轴以便广播。 `np.mean` 沿指定轴求算术平均。
        variance = np.mean((x - mean) ** 2, axis=1, keepdims=True)
        # 用标准差或均方根缩放输入，得到无量纲的归一化结果。 `np.sqrt` 对数组逐元素开平方。
        normalized = (x - mean) / np.sqrt(variance + epsilon)
    # 以上专用分支均不匹配时，执行这里约定的剩余算法分支。
    else:
        # RMSNorm 不减均值，只用均方根缩放原始输入。
        mean_square = np.mean(x**2, axis=1, keepdims=True)
        # 用标准差或均方根缩放输入，得到无量纲的归一化结果。 `np.sqrt` 对数组逐元素开平方。
        normalized = x / np.sqrt(mean_square + epsilon)
    # gamma 和 beta 按最后一个特征维广播，恢复可学习的缩放和平移。
    return normalized * gamma + beta
