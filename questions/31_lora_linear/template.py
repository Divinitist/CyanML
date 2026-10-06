import numpy as np


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
    pass
