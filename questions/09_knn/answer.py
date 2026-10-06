# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：按欧氏距离选择 K 个近邻，并完成分类投票或回归均值预测。
# API：stable argsort 保证同距离按原顺序；np.unique(..., return_counts=True) 同时返回标签及票数。

def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    k: int,
    task: str,
) -> np.ndarray:
    """
        Args:
            train_x: 形状为 (num_train, num_features) 的 float64 训练特征。
            train_y: 形状为 (num_train,) 的整数标签或浮点回归目标。
            query_x: 形状为 (num_queries, num_features) 的 float64 查询特征。
            k: 邻居数，满足 1 <= k <= num_train。
            task: 字符串 classification 或 regression。

        Returns:
            返回形状为 `(num_queries,)` 的预测数组。分类结果保持标签 dtype，回归结果为 float64。
        """
    # 插入长度为 1 的轴，利用广播一次生成所有“查询-训练”样本差值。
    distances = np.sqrt(np.sum((query_x[:, None, :] - train_x[None, :, :]) ** 2, axis=2))
    # stable 排序使距离相同时仍按训练集原顺序选邻居。
    neighbor_indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    # 二维索引为每个查询取出各自的 k 个目标值。
    neighbor_targets = train_y[neighbor_indices]
    # 回归任务无需投票，直接对邻居目标取均值并提前返回。
    if task == "regression":
        # 回归直接对邻居目标取算术平均。
        return np.mean(neighbor_targets, axis=1)
    # 把概率转换成离散预测类别。
    predictions = []
    # 逐个查询样本处理其 k 个邻居标签，完成分类多数投票。
    for targets in neighbor_targets:
        # unique 同时返回排序后的标签和票数；argmax 在平票时选择较小标签。
        labels, counts = np.unique(targets, return_counts=True)
        # `np.argmax(counts)` 取最高票标签的位置；`append` 把该查询的分类结果追加到列表末尾。
        predictions.append(labels[np.argmax(counts)])
    # `np.asarray(..., dtype=train_y.dtype)` 把预测列表转回与训练标签同 dtype 的数组。
    return np.asarray(predictions, dtype=train_y.dtype)
