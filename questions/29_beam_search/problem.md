# Beam Search

## 背景

Beam Search 在每一步保留若干累计分数最高的序列，比贪心搜索探索更多路径。本题用固定的一阶转移表代替神经网络，以隔离搜索算法本身。 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `beam_search`。序列从 start_token 开始，每一步用最后一个 token 对应的行取得下一 token 对数概率。已生成 end_token 的 beam 不再扩展。候选按长度惩罚后的分数降序保留 beam_width 个；完全并列时按 token 序列字典序升序。达到最大步数或所有 beam 完成后，返回同一规则下最优项。 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

- `transition_log_probs`：形状为 (vocab_size,vocab_size) 的一阶转移对数概率。
- `start_token / end_token`：合法的起始和终止 token 下标。
- `max_new_tokens`：最多追加的 token 数。
- `beam_width`：每轮保留的 beam 数。
- `length_penalty`：非负长度惩罚指数。

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

返回 `(tokens, score)`；tokens 包含 start_token，score 是最终累计对数概率除以生成长度的 length_penalty 次方。

## 数学定义

若生成长度（不含起始 token）为 L，排序分数为 $s(y)=\sum_t\log p(y_t|y_{t-1})/L^\alpha$；未生成 token 时分母按 1 处理。

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖贪心退化、多个 beam、提前结束、无结束路径、长度惩罚、分数并列和不同词表大小。本题误差标准为 `rtol=1e-10`、`atol=1e-12`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
