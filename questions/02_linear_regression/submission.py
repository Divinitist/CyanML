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
    pred = x @ weights + bias
    diff = pred - targets
    loss = np.mean(diff ** 2)
    # grad_w = 2 * diff * np.mean(x, axis=0) 没法广播，用 @ 可以自动广播
    grad_w = 2 * x.T @ diff / x.shape[0]
    grad_b = 2 * np.mean(diff)
    new_weights = weights - learning_rate * grad_w
    new_bias = bias - learning_rate * grad_b
    return (pred, loss, grad_w, grad_b, new_weights, new_bias)