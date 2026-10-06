import numpy as np


def softmax_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Args:
        logits: Shape (batch_size, num_classes), dtype float64.
        targets: Shape (batch_size,), containing integer class indices.

    Returns:
        probabilities: Shape (batch_size, num_classes), dtype float64.
        loss: Mean cross-entropy loss for the batch.
    """
    pass
