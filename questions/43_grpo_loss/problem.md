# GRPO Loss

## 背景

Group Relative Policy Optimization（GRPO）为同一 prompt 采样一组回答，用组内相对奖励构造优势，不依赖单独的 value model。策略目标采用类似 PPO 的 ratio clipping，并用参考策略 KL 项约束更新。

## 需求

实现 `grpo_loss`。先在每个 group 的 sample 维上标准化奖励，再对每条回答计算 clipped surrogate 与逐 token KL 惩罚。每条回答只平均 `mask=True` 的 token，最终对所有 group 和 sample 等权平均。输入保证每条回答至少有一个有效 token，不得修改输入数组。

## 输入

- `new_log_probs`：形状 `(groups, samples_per_group, sequence_length)` 的当前策略 token log-prob。
- `old_log_probs`：同形旧策略 token log-prob。
- `reference_log_probs`：同形参考策略 token log-prob。
- `rewards`：形状 `(groups, samples_per_group)` 的回答级奖励。
- `mask`：与 log-prob 同形的布尔有效 token 掩码。
- `clip_epsilon`：正的 ratio 裁剪范围。
- `kl_beta`：非负 KL 惩罚系数。
- `normalization_epsilon`：奖励标准化时加入方差的正数。

## 输出

返回 Python 浮点数，表示需要最小化的 GRPO loss。

## 数学定义

对每个组 $g$：

$$
A_{g,i}=\frac{R_{g,i}-\mu_g}{\sqrt{\frac{1}{G}\sum_j(R_{g,j}-\mu_g)^2+\epsilon}},
$$

$$
r_{g,i,t}=\exp(\log\pi_{new}-\log\pi_{old}),
$$

$$
D_{g,i,t}=\exp(\log\pi_{ref}-\log\pi_{new})-(\log\pi_{ref}-\log\pi_{new})-1.
$$

逐 token 目标为

$$
\min(rA,\operatorname{clip}(r,1-\epsilon_c,1+\epsilon_c)A)-\beta D,
$$

loss 是先按每条回答的有效 token 平均、再按回答平均所得目标的负值。

## 测试约定

共运行 100 组输入，覆盖不同组数、组大小、序列长度、变长 mask、相同奖励、正负相对优势、零 KL 系数和不同裁剪范围。误差标准为 `rtol=1e-11`、`atol=1e-12`。
