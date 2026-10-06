# 通用 KV Cache

## 背景

自回归推理若每一步都重新投影整个前缀，会重复计算历史 token 的 K/V。KV Cache 保存这些投影，使预填充和逐 token 解码共享同一个注意力接口。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `kv_cache_attention`。实现同时支持 MHA、MQA 和 GQA 的分块因果注意力。只把当前 x 投影出的 K/V 追加到缓存；查询头按连续分组复用 KV 头。当前分块第 i 个查询能看到全部历史缓存及当前块中不晚于 i 的键。cache 为 None 时视为空缓存，返回更新后的缓存。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `x`：当前分块，形状为 (chunk_length, model_dim)。
- `w_q`：形状为 (D,D) 的查询投影。
- `w_k / w_v`：形状为 (D,num_kv_heads*head_dim) 的 KV 投影。
- `w_o`：形状为 (D,D) 的输出投影。
- `num_query_heads / num_kv_heads`：查询头数与 KV 头数，前者可被后者整除。
- `cache_k / cache_v`：None 或形状为 (num_kv_heads,past_length,head_dim) 的历史缓存。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(output, new_cache_k, new_cache_v)`；output 为 `(chunk_length,D)`，新缓存长度为 past_length+chunk_length。

## 数学定义

查询头 h 使用 KV 头 $\lfloor h/(H_q/H_{kv})\rfloor$。当前块查询 i 的可见键满足 $j\le past+i$，注意力仍按 $1/\sqrt{d_h}$ 缩放。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖空缓存预填充、非空缓存解码、单 token 与多 token 块、MHA/MQA/GQA、不同历史长度和头维。本题误差标准为 `rtol=1e-8`、`atol=1e-10`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
