# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：在 NCHW 输入上执行带步幅和零填充的二维交叉相关。
# API：np.pad 扩展空间边界；np.tensordot 按指定三组轴同时做乘积求和。

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

    Each output position selects one strided input window. A tensor contraction
    sums the window against every output kernel across input channels and both
    spatial kernel axes. This is cross-correlation, matching neural frameworks.
    """
    # 输入和卷积核都使用 NCHW/输出通道优先的布局。
    batch_size, _, height, width = x.shape
    # 从卷积核形状解包输出通道数、核高和核宽；下划线忽略已知的输入通道维。
    out_channels, _, kernel_height, kernel_width = weight.shape
    # 只填充高、宽两维，批次和通道维保持不变。
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    # 整除公式给出卷积窗口能完整落下的空间位置数。
    output_height = (height + 2 * padding - kernel_height) // stride + 1
    # 用卷积尺寸公式计算输出宽度；整除会舍弃放不下完整卷积核的位置。
    output_width = (width + 2 * padding - kernel_width) // stride + 1
    # `np.empty` 分配 (N,C_out,H_out,W_out) 缓冲区但不初始化；下面双循环会写满每个空间位置。
    output = np.empty(
        (batch_size, out_channels, output_height, output_width),
        dtype=np.float64,
    )
    # 逐行遍历输出特征图的所有有效空间位置。
    for row in range(output_height):
        # 输出坐标乘 stride，得到其在填充后输入中的窗口起点。
        row_start = row * stride
        # 在当前输出行内逐列移动卷积窗口。
        for column in range(output_width):
            # 把输出列坐标乘步幅，换算成输入窗口的起始列。
            column_start = column * stride
            # 从填充后输入切出当前卷积核覆盖的 NCHW 子块。
            window = padded[
                :,
                :,
                row_start : row_start + kernel_height,
                column_start : column_start + kernel_width,
            ]
            # 同时沿输入通道和两个核空间轴求和，保留批次与输出通道轴。
            output[:, :, row, column] = np.tensordot(
                window,
                weight,
                axes=((1, 2, 3), (1, 2, 3)),
            ) + bias
    # 返回填满所有空间位置的 NCHW 卷积结果。
    return output
