import numpy as np


def best_decision_tree_split(
    x: np.ndarray,
    labels: np.ndarray,
    criterion: str,
) -> tuple[int, float, float]:
    """
        Args:
            x: 形状为 (num_samples, num_features) 的 float64 特征矩阵。
            labels: 形状为 (num_samples,) 的整数分类标签。
            criterion: 字符串 gini 或 entropy。

        Returns:
            返回 `(feature_index, threshold, gain)`，分别是最佳特征下标、浮点阈值和纯度增益。输入保证至少存在一个有效候选阈值。
        """
    pass
