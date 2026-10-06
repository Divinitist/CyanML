# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：依次执行输入投影、隐藏层激活和输出投影，得到两层 MLP 的前向结果。
# API：np.maximum(x, 0) 是逐元素 ReLU；np.tanh 是逐元素双曲正切激活。

def two_layer_mlp(
    x: np.ndarray,
    weight1: np.ndarray,
    bias1: np.ndarray,
    weight2: np.ndarray,
    bias2: np.ndarray,
    activation: str,
) -> np.ndarray:
    """
    Args:
        x: Shape (batch_size, input_size), dtype float64.
        weight1: Shape (input_size, hidden_size), dtype float64.
        bias1: Shape (hidden_size,), dtype float64.
        weight2: Shape (hidden_size, output_size), dtype float64.
        bias2: Shape (output_size,), dtype float64.
        activation: Either "relu" or "tanh".

    Returns:
        Output with shape (batch_size, output_size).

    Bias vectors broadcast across the batch dimension. Only the hidden layer
    receives a non-linearity; the second layer remains a raw linear output.
    """
    # 第一层把输入从 input_size 投影到 hidden_size，偏置按批广播。
    hidden = x @ weight1 + bias1
    # 选择 ReLU 时，对隐藏值逐元素截断负数。
    if activation == "relu":
        # maximum 逐元素截断负值，实现 ReLU。
        hidden = np.maximum(hidden, 0.0)
    # 以上专用分支均不匹配时，执行这里约定的剩余算法分支。
    else:
        # tanh 逐元素把隐藏值压缩到 (-1, 1)。
        hidden = np.tanh(hidden)
    # 输出层保持线性，不再施加激活函数。
    return hidden @ weight2 + bias2
