# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：统计 TP、FP、FN，并计算正类精确率与召回率的调和平均 F1。
# API：布尔数组的 & 是逐元素逻辑与；np.sum 会把 True/False 分别按 1/0 计数。

def binary_f1_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Count the three terms that determine the positive-class F1-score."""
    # 布尔与逐元素圈出 TP、FP、FN，sum 将 True 按 1 计数。
    true_positive = int(np.sum((predictions == 1) & (targets == 1)))
    # 统计预测为 1、真实为 0 的样本数。 `np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴；`int(...)` 把 NumPy 整数转换为普通 Python 整数。
    false_positive = int(np.sum((predictions == 1) & (targets == 0)))
    # 统计预测为 0、真实为 1 的样本数。 `np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴；`int(...)` 把 NumPy 整数转换为普通 Python 整数。
    false_negative = int(np.sum((predictions == 0) & (targets == 1)))
    # 将 precision/recall 的调和平均化简为 2TP/(2TP+FP+FN)。
    denominator = 2 * true_positive + false_positive + false_negative
    # 当 TP、FP、FN 全为零时分母为零，按题目约定返回 F1=0。
    if denominator == 0:
        # 没有真实正类也没有预测正类时，按题目约定返回 0。
        return 0.0
    # 按化简式 `2TP/(2TP+FP+FN)` 直接得到正类 F1。
    return float(2 * true_positive / denominator)
