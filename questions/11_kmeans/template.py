import numpy as np


def kmeans(
    x: np.ndarray,
    n_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
        Args:
            x: 形状为 (num_samples, num_features) 的 float64 数组。
            n_clusters: 簇数，不超过样本数。
            max_iterations: 最大更新轮数。
            tolerance: 非负收敛阈值。
            seed: 初始化使用的整数随机种子。

        Returns:
            返回 `(labels, centers)`；标签形状为 `(N,)`，中心形状为 `(K,D)`。中心编号由随机初始化顺序决定，不做额外排序。
        """
    pass
