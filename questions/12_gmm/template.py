import numpy as np


def gmm_em(
    x: np.ndarray,
    n_components: int,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Args:
            x: 形状为 (num_samples, num_features) 的 float64 数组。
            n_components: 混合分量数。
            max_iterations: EM 最大轮数。
            tolerance: 对数似然收敛阈值。
            regularization: 协方差对角线正则量。
            seed: 均值初始化的随机种子。

        Returns:
            返回 `(weights, means, covariances, responsibilities)`，形状分别为 `(K,)`、`(K,D)`、`(K,D,D)`、`(N,K)`。
        """
    pass
