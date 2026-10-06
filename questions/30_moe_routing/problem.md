# MoE 路由

## 背景

混合专家层为每个 token 只激活少量专家，从而在计算量近似不变时增加参数容量。路由器需要同时处理 Top-K 选择、门权重和专家容量。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `moe_route`。先计算 x@gate_weight 并对专家维做 softmax，再按概率稳定降序选择每个 token 的 top_k 专家，并只在选中项内重新 softmax。每个专家容量为 ceil(capacity_factor*num_tokens*top_k/num_experts)。按 token 顺序、槽位顺序接收路由，超容量项的 assignment 记为 -1 且不贡献输出；其余项通过对应专家的两层 ReLU MLP 并按门权重累加。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：形状为 (num_tokens, model_dim) 的 float64 token。
- `gate_weight`：形状为 (model_dim,num_experts) 的路由矩阵。
- `expert_w1`：形状为 (num_experts,model_dim,hidden_dim)。
- `expert_w2`：形状为 (num_experts,hidden_dim,model_dim)。
- `top_k`：每个 token 选择的专家数。
- `capacity_factor`：大于 0 的容量倍率。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(output, assignments, routing_weights)`；output 与 x 同形，后两者形状为 `(num_tokens,top_k)`，被丢弃的 assignment 为 -1。

## 数学定义

$p=softmax(xW_g)$，选中集合内 $\tilde p_e=e^{z_e}/\sum_{j\in TopK}e^{z_j}$；专家输出为 $ReLU(xW_{1,e})W_{2,e}$。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖Top-1/Top-2、不同专家数、容量充足与溢出、概率并列、不同 token/隐藏维度和稳定接收顺序。本题误差标准为 `rtol=1e-8`、`atol=1e-10`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
