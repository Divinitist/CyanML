import numpy as np


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
    pass
