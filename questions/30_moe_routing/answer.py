# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：为每个 token 选择 top-k 专家，在容量约束下加权合并专家输出。
# API：np.take_along_axis 按 top-k 下标从每行 logits 取值；np.ceil 向上取整专家容量。

def moe_route(
    x: np.ndarray,
    gate_weight: np.ndarray,
    expert_w1: np.ndarray,
    expert_w2: np.ndarray,
    top_k: int,
    capacity_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        x: 形状为 (num_tokens, model_dim) 的 float64 token。
        gate_weight: 形状为 (model_dim,num_experts) 的路由矩阵。
        expert_w1: 形状为 (num_experts,model_dim,hidden_dim)。
        expert_w2: 形状为 (num_experts,hidden_dim,model_dim)。
        top_k: 每个 token 选择的专家数。
        capacity_factor: 大于 0 的容量倍率。

    Returns:
        返回 `(output, assignments, routing_weights)`；output 与 x 同形，后两者形状为 `(num_tokens,top_k)`，被丢弃的 assignment 为 -1。
    """
    # 路由线性层为每个 token 对每位专家打分。
    logits = x @ gate_weight
    # stable 排序保证同分时专家下标较小者优先。
    order = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]
    # take_along_axis 按每行各自的 top-k 下标提取对应分数。
    selected = np.take_along_axis(logits, order, axis=1)
    # 对每个 token 选中的 top-k 专家分数做局部 Softmax。 `np.exp` 对数组逐元素取自然指数。
    routing_weights = np.exp(selected - selected.max(axis=1, keepdims=True))
    # 沿 top-k 专家轴归一化，使每个 token 的路由权重和为 1。
    routing_weights /= routing_weights.sum(axis=1, keepdims=True)
    # 复制专家下标，后续把因容量不足而丢弃的路由改为 -1。 `.copy()` 创建独立数组，避免后续原地修改影响来源。
    assignments = order.copy()
    # 从路由矩阵输出维读取专家总数。
    num_experts = gate_weight.shape[1]
    # 平均路由槽位乘容量系数并向上取整，得到每位专家的接收上限。
    capacity = int(np.ceil(capacity_factor * x.shape[0] * top_k / num_experts))
    # 记录每位专家已经接收的 token 数，用于执行容量限制。
    counts = np.zeros(num_experts, dtype=int)
    # `np.zeros_like(x)` 创建与 token 输入同形的零输出；每个接纳专家的加权结果会累加到对应行。
    output = np.zeros_like(x)
    # 逐个处理该集合中的元素，并把本轮结果写入对应输出位置。
    for token in range(x.shape[0]):
        # 逐个检查当前 token 的 top-k 路由槽位，决定是否接纳该专家。
        for slot in range(top_k):
            # 把 NumPy 整数专家下标转成 Python `int`，便于普通索引和计数。 `int(...)` 把 NumPy 整数转换为普通 Python 整数。
            expert = int(order[token, slot])
            # 专家已达到容量上限时丢弃这个路由槽，防止负载超过预设容量。
            if counts[expert] >= capacity:
                # 超容量路由标为 -1，且不把该专家输出加入 token。
                assignments[token, slot] = -1
                # 当前候选无需再扩展或已经超容量，跳到下一次循环。
                continue
            # 当前专家成功接收一个 token，因此把其容量计数加一。
            counts[expert] += 1
            # 当前 token 经过该专家的第一层权重，再用 `np.maximum(...,0)` 应用 ReLU。
            hidden = np.maximum(x[token] @ expert_w1[expert], 0.0)
            # 每位接收专家执行自己的两层 MLP，再按路由概率加权累加。
            output[token] += routing_weights[token, slot] * (hidden @ expert_w2[expert])
    # 返回 MoE 加权输出、实际路由下标和容量裁剪前的路由权重。
    return output, assignments, routing_weights
