# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：按位置把 Q/K 的相邻维度视作二维向量并旋转，从而注入相对位置信息。
# API：省略号切片 value[..., 0::2] 保留任意前导维；copy 防止覆盖仍待使用的输入分量。

def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    position_ids: np.ndarray,
    base: float = 10000.0,
    rotary_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Args:
            q / k: 形状为 (sequence_length, num_heads, head_dim) 的 float64 数组。
            position_ids: 形状为 (sequence_length,) 的位置数组。
            base: RoPE 频率基数。
            rotary_dim: None 或不超过 head_dim 的正偶数。

        Returns:
            返回旋转后的 `(q_rotated, k_rotated)`，两者形状不变且不修改输入。
        """
    # 未指定时旋转完整头维；否则只旋转前 rotary_dim 个通道。
    rotary_dim = q.shape[-1] if rotary_dim is None else rotary_dim
    # 每两个相邻通道共享一个随维度递减的旋转频率。
    frequencies = base ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    # 通过位置列向量与频率行向量广播，生成每个位置、每个维度对的旋转角。
    angles = position_ids[:, None] * frequencies[None, :]
    # 预计算所有 RoPE 角度的余弦，并插入头维以便广播到全部注意力头。
    cosine = np.cos(angles)[:, None, :]
    # 预计算所有 RoPE 角度的正弦，并插入头维以便广播到全部注意力头。
    sine = np.sin(angles)[:, None, :]

    # 定义 RoPE 二维旋转辅助函数，对偶数/奇数通道成对处理。
    def rotate(value):
        # copy 保留 rotary_dim 之后不参与旋转的通道，也避免修改调用方输入。
        result = value.copy()
        # 用步长为 2 的切片取出每对二维旋转向量的偶数分量。
        even = value[..., :rotary_dim:2]
        # 用步长为 2 的切片取出每对二维旋转向量的奇数分量。
        odd = value[..., 1:rotary_dim:2]
        # 对每对偶/奇通道应用标准二维旋转矩阵。
        result[..., :rotary_dim:2] = even * cosine - odd * sine
        # 按二维旋转公式写入这一半通道；省略号保留序列轴和头轴。
        result[..., 1:rotary_dim:2] = even * sine + odd * cosine
        # 返回旋转后的当前张量；`rotary_dim` 之后的通道因来自副本而保持不变。
        return result

    # 分别对 Q、K 应用同一组 RoPE 角度，并以二元组返回。
    return rotate(q), rotate(k)
