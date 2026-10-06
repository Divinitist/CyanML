# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：枚举特征中点，按 Gini 或熵的纯度增益选择最佳分类树划分。
# API：np.unique 去重并可返回计数；np.log2 逐元素计算以 2 为底的对数。

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
    # 定义局部杂质函数，统一计算 Gini 或信息熵，供所有候选划分复用。
    def impurity(values):
        # 空子节点没有类别分布；先单独处理可避免对空计数做除法。
        if values.size == 0:
            # 按约定把空节点杂质记为 0，使加权杂质公式仍可直接使用。
            return 0.0
        # unique 统计各类别频数，再换算为经验概率。
        _, counts = np.unique(values, return_counts=True)
        # 用各类别计数除以节点样本数，得到和为 1 的经验类别概率。
        probabilities = counts / values.size
        # Gini 衡量随机抽两个样本类别不同的概率；另一分支计算信息熵。
        if criterion == "gini":
            # Gini 杂质为 `1-sum(p_c^2)`；`float` 把 NumPy 标量转为 Python 标量。
            return float(1.0 - np.sum(probabilities ** 2))
        # 信息熵为 `-sum(p_c*log2(p_c))`；这里只遍历出现过的类别，所以不会计算 log2(0)。
        return float(-np.sum(probabilities * np.log2(probabilities)))

    # 父节点杂质固定，只需枚举子节点的加权杂质。
    parent = impurity(labels)
    # 初始化最佳划分记录；增益设为负无穷保证首个合法候选一定会被接纳。
    best_feature, best_threshold, best_gain = 0, 0.0, -np.inf
    # 逐列枚举每个输入特征，寻找全局最佳决策树划分。
    for feature in range(x.shape[1]):
        # `np.unique` 对当前特征列排序并去重，只保留会产生不同划分的取值。
        values = np.unique(x[:, feature])
        # 相邻不同取值的中点覆盖所有可能产生不同划分的阈值。
        thresholds = (values[:-1] + values[1:]) / 2.0
        # 逐个测试当前特征的候选中点阈值并计算纯度增益。
        for threshold in thresholds:
            # `<= threshold` 逐样本比较当前特征，得到进入左子节点的布尔掩码。
            left = x[:, feature] <= threshold
            # 用按位取反 `~` 得到阈值右侧样本掩码，与左侧掩码互补。
            right = ~left
            # 布尔数组的 mean 就是子节点样本占比，用它加权两边杂质。
            weighted = left.mean() * impurity(labels[left]) + right.mean() * impurity(labels[right])
            # 用父节点杂质减子节点加权杂质，得到当前划分的纯度增益。
            gain = parent - weighted
            # 只有严格改善才替换，因此完全平局时保留先枚举的特征和阈值。
            if gain > best_gain + 1e-15:
                # 记录新的最优特征、阈值和增益；`float` 把 NumPy 标量转成稳定的公开接口类型。
                best_feature, best_threshold, best_gain = feature, float(threshold), float(gain)
    # 返回纯度增益最大的特征、阈值和增益值。
    return best_feature, best_threshold, best_gain
