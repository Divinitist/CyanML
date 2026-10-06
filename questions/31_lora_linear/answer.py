# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：在冻结线性层旁加入低秩 LoRA 增量，并可对增量分支应用 inverted dropout。
# API：default_rng(seed).random(shape) 生成可复现均匀随机数；连续 @ 完成两次低秩投影。

def lora_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    alpha: float,
    dropout_p: float,
    training: bool,
    seed: int,
) -> np.ndarray:
    """
    Args:
        x: 形状为 (...,in_features) 的 float64 输入。
        weight: 形状为 (in_features,out_features) 的冻结权重。
        bias: None 或形状为 (out_features,) 的偏置。
        lora_a / lora_b: 形状为 (in_features,rank) 和 (rank,out_features) 的低秩矩阵。
        alpha: LoRA 缩放超参数，实际倍率为 alpha/rank。
        dropout_p / training / seed: 丢弃概率、训练标志和可复现随机种子。

    Returns:
        返回前导维度与 x 相同、末维为 out_features 的 float64 数组。
    """
    # 冻结主分支执行普通线性变换，bias 沿所有前导维广播。
    base = x @ weight
    # 只有线性层配置了偏置时才相加；一维 bias 会广播到所有前导样本维。
    if bias is not None:
        # 计算或更新冻结主线性层的输出。
        base = base + bias
    # 保存送入 LoRA 低秩分支的输入；训练时可能应用 dropout。
    adapted = x
    # 仅训练阶段且丢弃率非零时对 LoRA 分支采样 dropout；冻结主分支不受影响。
    if training and dropout_p > 0.0:
        # 只对 LoRA 分支做 dropout；固定 seed 使掩码可复现。
        keep = np.random.default_rng(seed).random(x.shape) >= dropout_p
        # 除以保留概率是 inverted dropout，使训练期望与推理一致。
        adapted = x * keep / (1.0 - dropout_p)
    # 从 LoRA A 矩阵第二维读取低秩维度，用于计算 `alpha/rank` 缩放。
    rank = lora_a.shape[1]
    # A 先降到 rank、B 再升到输出维，并按 alpha/rank 缩放增量。
    return base + (adapted @ lora_a @ lora_b) * (alpha / rank)
