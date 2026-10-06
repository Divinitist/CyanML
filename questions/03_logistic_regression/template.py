import numpy as np


def logistic_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), binary float64 labels.
        weights: Shape (num_features,), dtype float64.
        bias: Scalar bias.
        learning_rate: Scalar update rate.
        l2: L2 regularization coefficient.

    Returns:
        probabilities, loss, predictions, grad_weights, grad_bias,
        new_weights, new_bias.
    """
    pass
