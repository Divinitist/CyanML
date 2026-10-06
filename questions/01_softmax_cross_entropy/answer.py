# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：稳定地计算每个样本的 Softmax 概率和整批平均交叉熵。
# API：keepdims=True 保留被归约的轴以便广播；np.arange 构造逐行高级索引来取目标类别。

def softmax_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Args:
        logits: Shape (batch_size, num_classes), dtype float64.
        targets: Shape (batch_size,), containing integer class indices.

    Returns:
        probabilities: Shape (batch_size, num_classes), dtype float64.
        loss: Mean cross-entropy loss for the batch.

    Subtracting each row maximum keeps exponentiation finite without changing
    the softmax distribution. The loss is computed from log-sum-exp directly,
    rather than taking log(probabilities), so extremely unlikely target classes
    remain finite instead of underflowing to log(0).
    """
    # 每行减去最大 logit，既不改变 Softmax，又能避免后续 exp 上溢。
    row_max = np.max(logits, axis=1, keepdims=True)
    # 从每行 logit 中减去该行最大值，使最大元素变为 0，避免指数上溢且不改变 Softmax。
    shifted = logits - row_max
    # 把平移后的 logits 逐元素指数化，得到 Softmax 的未归一化权重。 `np.exp` 对数组逐元素取自然指数。
    exp_shifted = np.exp(shifted)
    # keepdims=True 保留列维度，使每行分母能广播回所有类别。
    probabilities = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    # 把平移量加回去，直接用 log-sum-exp 计算稳定的交叉熵。
    log_normalizer = np.log(np.sum(exp_shifted, axis=1)) + row_max[:, 0]
    # 行号与 targets 组成高级索引，逐样本取出真实类别的 logit。
    target_logits = logits[np.arange(logits.shape[0]), targets]
    # 对每个样本的 `logsumexp-target_logit` 求均值，得到批交叉熵。 `np.mean` 沿指定轴求算术平均；`float(...)` 把 NumPy 标量转换为公开接口要求的 Python 浮点数。
    loss = float(np.mean(log_normalizer - target_logits))
    # 同时返回稳定的类别概率和批平均交叉熵，顺序与公开接口一致。
    return probabilities, loss
