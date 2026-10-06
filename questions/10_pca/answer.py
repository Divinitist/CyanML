# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：通过中心化和奇异值分解得到主轴、低维表示及解释方差。
# API：np.linalg.svd(..., full_matrices=False) 返回紧凑 U/S/Vh；右奇异向量的行就是主成分。

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
    # PCA 必须先按特征中心化，否则第一主轴可能主要描述整体偏移。
    mean = np.mean(x, axis=0)
    # 逐特征减去训练均值，让 PCA 只分析围绕均值的变化。
    centered = x - mean
    # 紧凑 SVD 的 Vh 每一行都是一条按方差降序排列的主轴。
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    # 截取要求数量的主成分并复制，避免原地定向时改动 SVD 返回视图。 `.copy()` 创建独立数组，避免后续原地修改影响来源。
    components = components[:n_components].copy()
    # 逐个处理该集合中的元素，并把本轮结果写入对应输出位置。
    for row in range(components.shape[0]):
        # 奇异向量正负号等价；固定最大绝对值元素为正以保证结果可复现。
        pivot = np.argmax(np.abs(components[row]))
        # 若最大绝对值元素为负，就翻转整条主轴；SVD 轴的正负等价，这一步只固定可复现方向。
        if components[row, pivot] < 0:
            # 截取要求数量的主成分并复制，避免原地定向时改动 SVD 返回视图。
            components[row] *= -1.0
    # 将中心化样本投影到所选主轴，得到低维坐标。
    transformed = centered @ components.T
    # 奇异值平方除以 N-1，正是每条主轴上的样本方差。
    explained_variance = singular_values[:n_components] ** 2 / (x.shape[0] - 1)
    # 返回低维坐标、主轴、训练均值和各主轴解释方差。
    return transformed, components, mean, explained_variance
