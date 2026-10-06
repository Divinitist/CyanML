# Conv2D

## 背景

二维卷积从局部空间窗口提取特征，并在多个输入、输出通道之间组合信息。本题实现深度学习框架常用的互相关形式，不翻转卷积核。

## 需求

实现 `conv2d(x, weight, bias, stride, padding)`，数据布局统一使用 NCHW。

## 输入

- `x`：`(batch_size, in_channels, height, width)` 的 `float64` 数组；
- `weight`：`(out_channels, in_channels, kernel_height, kernel_width)`；
- `bias`：`(out_channels,)`；
- `stride`、`padding`：非负整数，其中 `stride` 大于零。

输入尺寸均保证输出空间尺寸为正整数。不需要处理 dilation 或 groups。

## 输出

返回 `(batch_size, out_channels, output_height, output_width)` 的 `float64` 数组，其中：

$$H_{out}=\left\lfloor\frac{H+2P-K_H}{S}\right\rfloor+1$$

宽度同理。

## 测试约定

100 组输入覆盖单/多 batch、单/多通道、非方形输入与卷积核、不同 stride、padding、1×1 卷积及不同数值尺度。误差为 `rtol=1e-10`、`atol=1e-12`。
