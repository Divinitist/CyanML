import numpy as np


def normalization(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mode: str,
    epsilon: float,
) -> np.ndarray:
    """
    Args:
        x: Shape (batch_size, feature_size), dtype float64.
        gamma: Shape (feature_size,), dtype float64.
        beta: Shape (feature_size,), dtype float64.
        mode: One of "batch", "layer", or "rms".
        epsilon: Positive numerical stabilizer.

    Returns:
        Normalized values with the same shape as x.
    """
    # y = gamma x hatx + beta
    # hatx = (x - mu) / sqrt(sg^2+eps)
    if mode == 'batch':
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.mean((x-mean) ** 2, axis=0, keepdims=True)
        hatx = (x - mean) / np.sqrt(var + epsilon)
    elif mode == 'layer':
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x-mean) ** 2, axis=-1, keepdims=True)
        hatx = (x - mean) / np.sqrt(var + epsilon)
    else:
        var = np.mean(x ** 2, axis=-1, keepdims=True)
        hatx = x / np.sqrt(var + epsilon)
    return gamma * hatx + beta
        
        
