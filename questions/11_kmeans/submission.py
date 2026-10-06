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

        取中心，算每个点离各中心的距离，取每个点对应中心编号作为标记，聚类求和除以模，得到新中心，迭代直至重心移动距离<= tol
        """
    N, D = x.shape
    K = n_clusters
    c = x[np.random.default_rng(seed).choice(N, size=K, replace=False)].copy()
    for _ in range(max_iterations):
        d = np.sqrt(np.sum((x[:,None] - c[None])**2, axis=-1)) # (N,D)
        labels = np.argmin(d, axis=-1) # (N,)
        nc = c.copy()
        for k in range(K):
            clus = x[labels == k] # (?,D)
            if clus.size:
                nc[k] = np.mean(clus, axis=0)
        maxmov = np.max(np.sqrt(np.sum((nc - c)**2, axis=-1)), axis=0)
        c = nc
        if maxmov <= tolerance: break
    return np.argmin(np.sqrt(np.sum((x[:,None]-c[None])**2, axis=-1)), axis=-1), c
