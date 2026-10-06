# Transformer FFN

## 背景

Transformer 的逐位置前馈网络独立处理每个 token，在模型维度和更宽的隐藏维度之间变换。门控变体把内容分支与激活后的门分支逐元素相乘。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `transformer_ffn`。支持 relu、gelu、glu、geglu 和 swiglu。relu/gelu 只使用 up 分支；三个门控模式还使用 gate 分支，其中 glu 的门激活为 sigmoid，geglu 为 GELU，swiglu 为 SiLU。GELU 采用给定的 tanh 近似式。最后统一通过 down 投影。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：形状为 (..., model_dim) 的 float64 输入。
- `w_up / b_up`：形状为 (model_dim, hidden_dim) 和 (hidden_dim,) 的内容投影。
- `w_down / b_down`：形状为 (hidden_dim, model_dim) 和 (model_dim,) 的输出投影。
- `activation`：relu、gelu、glu、geglu 或 swiglu。
- `w_gate / b_gate`：门控模式使用的 (model_dim, hidden_dim) 和 (hidden_dim,)；非门控模式为 None。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回与 x 前导维度相同、末维为 model_dim 的 float64 数组。

## 数学定义

$GELU(t)=0.5t[1+\tanh(\sqrt{2/\pi}(t+0.044715t^3))]$，$SiLU(t)=t/(1+e^{-t})$；门控隐藏层为 $up(x)\odot act(gate(x))$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖五种激活、二维和三维输入、不同模型/隐藏维度、零偏置、正负大值和门控分支。本题误差标准为 `rtol=1e-9`、`atol=1e-11`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
