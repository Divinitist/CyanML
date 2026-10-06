# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：把模型维拆成多个头，执行缩放点积注意力后拼接并输出投影。
# API：reshape+transpose 重排为 (头, 序列, 头维)；np.where 用 -inf 屏蔽不可见位置。

def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (sequence_length, model_dim) 的 float64 序列。
            w_q / w_k / w_v / w_o: 四个形状均为 (model_dim, model_dim) 的投影矩阵。
            num_heads: 注意力头数，能整除 model_dim。
            mask: None 或形状为 (L,L) 的布尔数组。

        Returns:
            返回形状为 `(L, model_dim)` 的注意力输出。
        """
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    length, model_dim = x.shape
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = model_dim // num_heads
    # 先线性投影，再重排成 (num_heads, length, head_dim) 以并行计算各头。
    q = (x @ w_q).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Key，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    k = (x @ w_k).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    v = (x @ w_v).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    # K 交换最后两轴后与 Q 点积；除以 sqrt(head_dim) 控制分数尺度。
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    # 只有调用方提供可见性 mask 时才屏蔽注意力位置；None 表示全可见。
    if mask is not None:
        # 不可见位置置为负无穷，经过 Softmax 后权重严格为零。
        scores = np.where(mask[None, :, :], scores, -np.inf)
    # 减去每行最大值，稳定计算注意力 Softmax。
    maximum = np.max(scores, axis=-1, keepdims=True)
    # 先对平移后的注意力分数逐元素取指数，得到未归一化注意力权重。 `np.exp` 对数组逐元素取自然指数。
    weights = np.exp(scores - maximum)
    # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。 `np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴。
    weights /= np.sum(weights, axis=-1, keepdims=True)
    # 加权 V 后把头维移回中间并拼接，恢复 (length, model_dim)。
    context = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim)
    # 用 `w_o` 对拼接后的多头上下文做输出投影，结果恢复模型维度。
    return context @ w_o
