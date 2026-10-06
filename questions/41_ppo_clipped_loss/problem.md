# PPO Clipped Policy Loss

## 背景

PPO 用概率比率衡量新旧策略对同一动作的相对偏好，并通过裁剪代理目标限制单次策略更新幅度。

## 需求

实现 `ppo_clipped_loss`，计算不包含 value loss 和 entropy bonus 的 clipped policy loss。输入数组形状相同，可为一维或多维；最终对所有元素取平均。不得修改输入数组。

## 输入

- `new_log_probs`：新策略所选动作的 `float64` 对数概率。
- `old_log_probs`：旧策略对应的 `float64` 对数概率，与前者同形。
- `advantages`：同形 `float64` 优势。
- `clip_epsilon`：正的裁剪范围参数。

## 输出

返回 Python 浮点数，表示需要最小化的 policy loss。

## 数学定义

$$
r_i=\exp(\log\pi_{new,i}-\log\pi_{old,i}),
$$

$$
L=-\frac{1}{N}\sum_i\min\left(r_iA_i,\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)A_i\right).
$$

## 测试约定

共运行 100 组输入，覆盖正负及零优势、比率位于和超出裁剪区间、不同张量形状、零更新及不同 `clip_epsilon`。误差标准为 `rtol=1e-12`、`atol=1e-12`。
