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
    pass
