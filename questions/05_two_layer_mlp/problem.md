# 两层 MLP

## 背景

多层感知机通过线性变换和非线性激活组合特征。本题实现一个包含隐藏层和输出层的完整前向传播。

## 需求

实现 `two_layer_mlp(x, weight1, bias1, weight2, bias2, activation)`。第一层线性输出经过指定激活函数，再送入第二层线性变换。按照施工规范，本题只考 forward，不要求手写 backward。

## 输入

- `x`：形状 `(batch_size, input_size)` 的 `float64` 数组；
- `weight1`：形状 `(input_size, hidden_size)`；
- `bias1`：形状 `(hidden_size,)`；
- `weight2`：形状 `(hidden_size, output_size)`；
- `bias2`：形状 `(output_size,)`；
- `activation`：字符串，取值为 `"relu"` 或 `"tanh"`。

## 输出

返回形状 `(batch_size, output_size)` 的 `float64` 数组。第二层之后不附加激活函数。

## 测试约定

100 组合法输入覆盖两种激活、单样本、不同层宽、零参数、负输入和不同数值尺度。误差为 `rtol=1e-10`、`atol=1e-12`。
