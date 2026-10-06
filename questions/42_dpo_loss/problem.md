# DPO Loss

## 背景

Direct Preference Optimization（DPO）直接利用 chosen/rejected 偏好对优化策略。它比较策略模型与参考模型的偏好 log-ratio，无需显式训练奖励模型。

## 需求

实现 `dpo_loss`。四个 log-prob 数组形状相同，每个元素代表一组偏好样本的整段序列 log-prob。返回 batch mean loss。计算需要在较大正负 margin 下保持数值稳定，不得修改输入数组。

## 输入

- `chosen_log_probs`：当前策略对 chosen 序列的 `float64` log-prob。
- `rejected_log_probs`：当前策略对 rejected 序列的同形 log-prob。
- `reference_chosen_log_probs`：参考策略的 chosen log-prob。
- `reference_rejected_log_probs`：参考策略的 rejected log-prob。
- `beta`：正的偏好温度系数。

## 输出

返回 Python 浮点数，表示所有偏好样本的平均 DPO loss。

## 数学定义

$$
z_i=\beta\left[(\log\pi(y_w|x)-\log\pi(y_l|x))-(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x))\right],
$$

$$
L=-\frac{1}{N}\sum_i\log\sigma(z_i).
$$

## 测试约定

共运行 100 组输入，覆盖策略优于或劣于参考策略、零 margin、不同 batch 形状、多个 `beta` 以及大绝对值 log-prob。误差标准为 `rtol=1e-12`、`atol=1e-12`。
