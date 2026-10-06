# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：实现普通 ReLU/GELU FFN，以及 GLU、GEGLU、SwiGLU 三种门控 FFN。
# API：np.tanh 用于 GELU 近似；广播让一维 bias 自动加到任意前导维的最后一轴。

def transformer_ffn(
    x: np.ndarray,
    w_up: np.ndarray,
    b_up: np.ndarray,
    w_down: np.ndarray,
    b_down: np.ndarray,
    activation: str,
    w_gate: np.ndarray | None = None,
    b_gate: np.ndarray | None = None,
) -> np.ndarray:
    """
    Args:
        x: 形状为 (..., model_dim) 的 float64 输入。
        w_up / b_up: 形状为 (model_dim, hidden_dim) 和 (hidden_dim,) 的内容投影。
        w_down / b_down: 形状为 (hidden_dim, model_dim) 和 (model_dim,) 的输出投影。
        activation: relu、gelu、glu、geglu 或 swiglu。
        w_gate / b_gate: 门控模式使用的 (model_dim, hidden_dim) 和 (hidden_dim,)；非门控模式为 None。

    Returns:
        返回与 x 前导维度相同、末维为 model_dim 的 float64 数组。
    """
    # 内容分支把最后一维从 model_dim 扩展到 hidden_dim。
    up = x @ w_up + b_up

    # 定义 GELU 的 tanh 近似，避免依赖额外深度学习框架。
    def gelu(value):
        # 使用 Transformer 中常见的 tanh 近似式计算 GELU。
        return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * value ** 3)))

    # 选择 ReLU 时，对隐藏值逐元素截断负数。
    if activation == "relu":
        # `np.maximum(up, 0)` 逐元素把负值截成 0，得到 ReLU 隐藏层。
        hidden = np.maximum(up, 0.0)
    # 选择 GELU 时，使用上面定义的平滑激活近似。
    elif activation == "gelu":
        # 对内容分支直接应用 GELU，得到平滑的非门控隐藏层。
        hidden = gelu(up)
    # 以上专用分支均不匹配时，执行这里约定的剩余算法分支。
    else:
        # 门控变体额外生成同形 gate，再根据类型施加 Sigmoid/GELU/SiLU。
        gate = x @ w_gate + b_gate
        # GLU 使用 Sigmoid 作为门函数。
        if activation == "glu":
            # 用 `1/(1+exp(-x))` 把门值压到 (0,1)，实现标准 GLU 的 Sigmoid 门。
            gate = 1.0 / (1.0 + np.exp(-gate))
        # GEGLU 使用 GELU 作为门函数。
        elif activation == "geglu":
            # 对门分支应用 GELU，形成 GEGLU 的平滑门值。
            gate = gelu(gate)
        # 以上专用分支均不匹配时，执行这里约定的剩余算法分支。
        else:
            # 计算 `x*sigmoid(x)` 得到 SiLU 门值，这是 SwiGLU 的门函数。
            gate = gate / (1.0 + np.exp(-gate))
        # 逐元素相乘让门值控制内容分支每个隐藏单元的通过量。
        hidden = up * gate
    # 下投影把 hidden_dim 映射回 model_dim。
    return hidden @ w_down + b_down
