import numpy as np


def softmax_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), integer class indices.
        weights: Shape (num_features, num_classes), dtype float64.
        bias: Shape (num_classes,), dtype float64.
        learning_rate: Scalar update rate.
        l2: L2 regularization coefficient.

    Returns:
        probabilities, loss, predictions, grad_weights, grad_bias,
        new_weights, new_bias.
    """
    pass
