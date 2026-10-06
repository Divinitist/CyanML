import numpy as np


def two_layer_mlp(
    x: np.ndarray,
    weight1: np.ndarray,
    bias1: np.ndarray,
    weight2: np.ndarray,
    bias2: np.ndarray,
    activation: str,
) -> np.ndarray:
    """
    Args:
        x: Shape (batch_size, input_size), dtype float64.
        weight1: Shape (input_size, hidden_size), dtype float64.
        bias1: Shape (hidden_size,), dtype float64.
        weight2: Shape (hidden_size, output_size), dtype float64.
        bias2: Shape (output_size,), dtype float64.
        activation: Either "relu" or "tanh".

    Returns:
        Output with shape (batch_size, output_size).
    """
    if activation == 'relu':
        f = lambda x: np.maximum(0.0, x)
    elif activation == 'tanh':
        f = lambda x: np.tanh(x)
    else: return None
    return (f(x @ weight1 + bias1)) @ weight2 + bias2
