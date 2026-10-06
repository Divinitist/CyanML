import numpy as np


def conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int,
    padding: int,
) -> np.ndarray:
    """
    Args:
        x: Shape (batch_size, in_channels, height, width), dtype float64.
        weight: Shape (out_channels, in_channels, kernel_height, kernel_width).
        bias: Shape (out_channels,), dtype float64.
        stride: Integer spatial stride.
        padding: Symmetric zero padding on every spatial edge.

    Returns:
        Convolution output in NCHW layout.
    """
    pass
