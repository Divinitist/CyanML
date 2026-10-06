# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：仅关注当前位置及其前方固定窗口，计算局部因果多头注意力。
# API：np.arange 配合二维广播生成位置比较矩阵；位运算 & 合并两个布尔条件。

def sliding_window_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    window_size: int,
) -> np.ndarray:
    """
        Args:
            x: 形状为 (L,D) 的 float64 序列。
            w_q / w_k / w_v / w_o: 四个 (D,D) 投影矩阵。
            num_heads: 能整除 D 的注意力头数。
            window_size: 正整数窗口宽度，包含当前位置。

        Returns:
            返回形状为 `(L,D)` 的局部因果注意力结果。
        """
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    length, model_dim = x.shape
    # 用模型维度除以头数，得到每个注意力头负责的特征数。
    head_dim = model_dim // num_heads
    # 将 Q/K/V 统一整理为 (head, position, head_dim)。
    q = (x @ w_q).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Key，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    k = (x @ w_k).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    # 线性投影得到 Value，`reshape` 拆出头维，`transpose` 把布局改为 (head, sequence, head_dim)。
    v = (x @ w_v).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    # 每个头计算 `Q @ K^T` 得到位置两两相似度，再除以 `sqrt(head_dim)` 防止 Softmax 饱和。
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    # 两次广播形成 query-key 位置矩阵：既不能看未来，也不能超出左侧窗口。
    positions = np.arange(length)
    # 构造布尔可见性矩阵，True 表示该 Query 可以读取对应 Key。
    mask = (positions[None, :] <= positions[:, None]) & (positions[None, :] >= positions[:, None] - window_size + 1)
    # 被屏蔽分数置为负无穷，使对应 Softmax 权重为零。
    scores = np.where(mask[None, :, :], scores, -np.inf)
    # 减最大值后归一化，避免指数运算溢出。
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    # 沿 Key 轴除以每行指数和，把未归一化权重变成注意力概率。
    weights /= weights.sum(axis=-1, keepdims=True)
    # 对 V 加权后拼接各头，恢复模型维并做输出投影。
    joined = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim)
    # 输出投影把局部注意力各头拼接结果映射回模型维。
    return joined @ w_o
