import numpy as np


def linear_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), dtype float64.
        weights: Shape (num_features,), dtype float64.
        bias: Scalar bias before the update.
        learning_rate: Scalar gradient-descent learning rate.

    Returns:
        predictions, loss, grad_weights, grad_bias, new_weights, new_bias.
    """
    pass
