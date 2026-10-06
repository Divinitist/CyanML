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
    # 中心化
    mean = np.mean(x, axis=0)
    x = x - mean
    # 分解，取右奇异值
    _, s, v = np.linalg.svd(x, full_matrices=False)
    # 取主成分
    v = v[:n_components].copy()
    # 把最大的变成正数
    for i in range(v.shape[0]):
        tar = np.argmax(np.abs(v[i]))
        if v[i][tar] < 0: v[i] *= -1
    ans = x @ v.T
    evar = s[:n_components] ** 2 / (x.shape[0] - 1)
    return ans, v, mean, evar
    
