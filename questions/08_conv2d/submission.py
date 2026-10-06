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
    B, I, H, W = x.shape
    O, _, KH, KW = weight.shape
    P = padding
    S = stride
    # 每一维的左右 pad 量
    px = np.pad(x, ((0,0),(0,0),(P,P),(P,P)))
    OH = (H + 2 * P - KH) // S + 1
    OW = (W + 2 * P - KW) // S + 1
    o = np.empty((B, O, OH, OW))
    for row in range(OH):
        rowl = row * S
        for col in range(OW):
            coll = col * S
            window = px[:,:,rowl:rowl+KH,coll:coll+KW]
            o[:,:,row,col] = (window[:,None] * weight[None]).sum(axis=(-3,-2,-1)) + bias
    return o
    
