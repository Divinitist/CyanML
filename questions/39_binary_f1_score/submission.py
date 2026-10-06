import numpy as np


def binary_f1_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Args:
        predictions: Shape (num_samples,), binary predicted labels.
        targets: Shape (num_samples,), binary target labels.

    Returns:
        Positive-class F1-score as a Python float.
    """
    TP = np.sum(predictions * targets)
    FP = np.sum(predictions * (targets==0))
    FN = np.sum((predictions==0) * targets)
    return 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0
