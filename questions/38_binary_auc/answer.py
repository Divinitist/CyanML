# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：用 Mann-Whitney 秩统计计算二分类 ROC-AUC，并正确处理同分样本。
# API：stable argsort 保持同分项原顺序；连续相等分数共享其一基排名区间的平均秩。

def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Mann-Whitney rank statistic behind ROC-AUC.

    A stable sort groups equal scores together. Every item in a tie group gets
    the group's average one-based rank, which gives each positive-negative tie
    the required half credit. The rank-sum identity then avoids constructing
    the full positive-by-negative comparison matrix.
    """
    # 稳定升序排序后，order 可把排好序的位置映射回原样本下标。
    order = np.argsort(scores, kind="mergesort")
    # 按升序下标重排分数，使相同分数连续出现并可成组处理。
    sorted_scores = scores[order]
    # 分配每个原样本的浮点秩缓冲区，平分时需要写入平均秩。 `np.empty` 只分配指定形状的内存，使用前必须把每个位置写满。
    ranks = np.empty(scores.size, dtype=np.float64)

    # 记录当前同分组在排序数组中的左边界。
    start = 0
    # 条件 `start < scores.size` 成立时继续循环。
    while start < scores.size:
        # 向右扫描出当前所有相同分数组成的连续 tie group。
        end = start + 1
        # 条件 `end < scores.size and sorted_scores[end] ==…` 成立时继续循环。
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            # 右边界加一，继续吞并与当前起点分数完全相同的样本。
            end += 1
        # 一基排名从 start+1 到 end，其平均值写回这组样本的原位置。
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        # 记录当前同分组在排序数组中的左边界。
        start = end

    # 与标签 1 逐元素比较，得到选取所有正样本秩的布尔掩码。
    positive = labels == 1
    # 统计正样本数，作为 AUC 的正负样本对分母之一。 `np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴；`int(...)` 把 NumPy 整数转换为普通 Python 整数。
    positive_count = int(np.sum(positive))
    # 用总样本数减正样本数，得到负样本数。
    negative_count = labels.size - positive_count
    # Mann-Whitney 公式从正样本秩和扣掉其最小可能秩和。
    rank_sum = float(np.sum(ranks[positive]))
    # 从正样本秩和中扣除连续最小秩和，得到 Mann-Whitney U 统计量。
    statistic = rank_sum - positive_count * (positive_count + 1) / 2.0
    # 除以正负样本对总数，得到随机正样本得分高于负样本的概率。
    return float(statistic / (positive_count * negative_count))
