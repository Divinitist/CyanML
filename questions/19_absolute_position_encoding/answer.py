# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：生成不同频率的正弦/余弦绝对位置编码并加到 token 表示上。
# API：0::2 和 1::2 是步长为 2 的切片，分别写入偶数维与奇数维。

def add_sinusoidal_position_encoding(
    x: np.ndarray,
    start_position: int = 0,
    base: float = 10000.0,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (sequence_length, model_dim) 的 float64 数组。
            start_position: 第一个 token 的非负绝对位置。
            base: 控制频率跨度的正浮点数。

        Returns:
            返回与 x 同形状的 `x + position_encoding`。
        """
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    length, model_dim = x.shape
    # 列向量位置与行向量频率广播，生成 (length, ceil(model_dim/2)) 的角度表。
    positions = (start_position + np.arange(length, dtype=np.float64))[:, None]
    # 生成 0、2、4… 的偶数通道下标，每对正余弦通道共享一个频率。
    even_indices = np.arange(0, model_dim, 2)
    # 维度越高分母越大，因此正弦波变化越慢。
    angles = positions / (base ** (even_indices / model_dim))[None, :]
    # 创建与输入同形的零矩阵，随后分别填入偶数维 sin 和奇数维 cos。 `np.zeros_like` 创建与参照数组同形、同 dtype 的全零数组。
    encoding = np.zeros_like(x, dtype=np.float64)
    # 偶数通道写入 sin，奇数通道写入同频率的 cos。
    encoding[:, 0::2] = np.sin(angles)
    # model_dim 为奇数时，奇数通道比偶数通道少一个，故截断 angles。
    odd_count = encoding[:, 1::2].shape[1]
    # 把余弦值写入奇数通道；切掉多余角度以兼容奇数 model_dim。
    encoding[:, 1::2] = np.cos(angles[:, :odd_count])
    # 把位置编码逐元素加到 token 表示上；加法不改变输入形状。
    return x + encoding
