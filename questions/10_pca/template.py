import numpy as np


def pca(
    x: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Args:
            x: 形状为 (num_samples, num_features) 的 float64 数据矩阵，样本数至少为 2。
            n_components: 保留的主成分数，不超过样本数和特征数。

        Returns:
            依次返回 `transformed`、`components`、`mean`、`explained_variance`。形状分别为 `(N,C)`、`(C,D)`、`(D,)` 和 `(C,)`。
        """
    pass
