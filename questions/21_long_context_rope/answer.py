# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：通过线性位置缩放或 NTK 基数缩放，把 RoPE 扩展到更长上下文。
# API：np.empty_like 只按输入形状和 dtype 分配结果；随后偶/奇切片会填满全部元素。

def apply_scaled_rope(
    q: np.ndarray,
    k: np.ndarray,
    position_ids: np.ndarray,
    scale: float,
    method: str,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Args:
            q / k: 形状为 (L,H,d) 的 float64 数组。
            position_ids: 形状为 (L,) 的绝对位置。
            scale: 大于等于 1 的上下文扩展倍率。
            method: 字符串 linear 或 ntk。
            base: 原始 RoPE 基数。

        Returns:
            返回缩放旋转后的 Q/K 二元组，形状与输入一致。
        """
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = q.shape[-1]
    # linear 模式通过缩小位置编号减慢旋转，频率基数保持原值。
    if method == "linear":
        # 线性缩放把位置除以扩展倍率，相当于减慢所有旋转频率。
        positions = position_ids / scale
        # 确定长上下文 RoPE 使用的频率基数。
        adjusted_base = base
    # 以上专用分支均不匹配时，执行这里约定的剩余算法分支。
    else:
        # NTK 缩放保留位置，改大频率基数以更多调整低频维度。
        positions = position_ids
        # 确定长上下文 RoPE 使用的频率基数。
        adjusted_base = base * scale ** (head_dim / (head_dim - 2))
    # 每一对偶/奇通道共享同一个角频率。
    frequencies = adjusted_base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    # 通过位置列向量与频率行向量广播，生成每个位置、每个维度对的旋转角。
    angles = positions[:, None] * frequencies[None, :]
    # 预计算所有 RoPE 角度的余弦，并插入头维以便广播到全部注意力头。
    cosine = np.cos(angles)[:, None, :]
    # 预计算所有 RoPE 角度的正弦，并插入头维以便广播到全部注意力头。
    sine = np.sin(angles)[:, None, :]

    # 定义 RoPE 二维旋转辅助函数，对偶数/奇数通道成对处理。
    def rotate(value):
        # empty_like 分配同形状结果；下面两个切片会覆盖全部偶/奇通道。
        result = np.empty_like(value)
        # 用步长为 2 的切片同时解包偶数和奇数通道，组成二维旋转对。
        even, odd = value[..., 0::2], value[..., 1::2]
        # 对每对通道应用二维旋转，前导的序列和头维由广播保留。
        result[..., 0::2] = even * cosine - odd * sine
        # 按二维旋转公式写入这一半通道；省略号保留序列轴和头轴。
        result[..., 1::2] = even * sine + odd * cosine
        # 返回当前张量的完整旋转结果；偶数和奇数切片已覆盖最后一维所有通道。
        return result

    # 分别旋转 Q、K；两者共享缩放后的频率和位置。
    return rotate(q), rotate(k)
