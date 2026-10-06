# Dropout

## 背景

Dropout 在训练时随机屏蔽激活以减少共适应，推理时保持所有激活。Inverted Dropout 在训练阶段完成缩放，因此推理阶段无需额外乘系数。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `dropout`。training=True 且 probability>0 时，用 default_rng(seed).random(x.shape) >= probability 生成布尔保留掩码，输出 x*mask/(1-probability)。推理模式或 probability=0 时返回 x 的副本和全 True 掩码。不得修改 x。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：任意形状的 float64 输入数组。
- `probability`：满足 0<=p<1 的丢弃概率。
- `training`：True 表示训练模式，False 表示推理模式。
- `seed`：生成掩码的整数随机种子。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(output, keep_mask)`；output 与 x 同形，keep_mask 是同形布尔数组。

## 数学定义

训练时 $y_i=x_i m_i/(1-p)$，$m_i\sim Bernoulli(1-p)$，因此对随机掩码取期望有 $E[y_i]=x_i$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖标量风格小数组及多维数组、p=0、高丢弃率、训练/推理、全零输入、不同随机种子和复现性。本题误差标准为 `rtol=1e-12`、`atol=1e-12`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
