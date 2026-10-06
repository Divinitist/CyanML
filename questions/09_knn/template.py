import numpy as np


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
    pass
