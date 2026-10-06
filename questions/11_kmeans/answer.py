# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：交替执行最近中心分配与簇均值更新，直至中心移动量收敛。
# API：default_rng(seed) 创建独立可复现生成器；choice(..., replace=False) 无放回选初始中心。

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
    # 独立 RNG 配合固定 seed，使初始化不受全局随机状态影响且可复现。
    rng = np.random.default_rng(seed)
    # replace=False 保证初始中心来自互不重复的训练样本。
    centers = x[rng.choice(x.shape[0], size=n_clusters, replace=False)].copy()
    # 最多执行 `max_iterations` 轮迭代；循环内部满足收敛条件时会提前 `break`。
    for _ in range(max_iterations):
        # 广播得到每个样本到每个中心的平方欧氏距离，无需开方即可比较。
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        # 为每个样本选择距离最近的中心编号。 `np.argmin` 返回指定轴上最小元素的下标。
        labels = np.argmin(distances, axis=1)
        # 复制旧中心作为本轮候选值，使空簇可以安全保留旧中心。 `.copy()` 创建独立数组，避免后续原地修改影响来源。
        new_centers = centers.copy()
        # 逐个簇收集成员并更新中心；空簇会保留旧中心。
        for cluster in range(n_clusters):
            # 用布尔索引取出当前簇包含的全部样本。
            members = x[labels == cluster]
            # 空簇保留旧中心，避免对空数组求均值产生 NaN。
            if members.size:
                # 用当前簇成员的逐特征均值更新该簇中心；`axis=0` 保留特征维。 `np.mean` 沿指定轴求算术平均。
                new_centers[cluster] = np.mean(members, axis=0)
        # 计算每个中心本轮移动的欧氏距离，用于判断收敛。 `np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴；`np.sqrt` 对数组逐元素开平方。
        movement = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))
        # 接受本轮更新后的中心，作为下一轮分配依据。
        centers = new_centers
        # 所有中心的最大移动量不超过阈值时提前停止。
        if np.max(movement) <= tolerance:
            # 收敛或终止条件已经满足，提前退出迭代以避免无效计算。
            break
    # 用最终中心重新分配一次，保证返回标签与返回中心严格对应。
    labels = np.argmin(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
    # 返回最终中心对应的样本标签以及收敛后的中心坐标。
    return labels, centers
