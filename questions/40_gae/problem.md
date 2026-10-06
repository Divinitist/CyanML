# 广义优势估计 GAE

## 背景

广义优势估计（Generalized Advantage Estimation）用参数 $\lambda$ 在低方差的一步 TD 估计与低偏差的长回报估计之间折中，是 PPO 等策略梯度算法的常用组件。

## 需求

实现 `generalized_advantage_estimation`，沿时间维从后向前递推优势。`dones[t]=True` 表示执行第 `t` 步后轨迹终止，此时不得 bootstrap 到 `values[t+1]`，也不得把下一步优势传回来。不得修改输入数组。

## 输入

- `rewards`：形状 `(time_steps,)` 的 `float64` 即时奖励。
- `values`：形状 `(time_steps + 1,)` 的 `float64` 状态价值，末项是最后状态的 bootstrap value。
- `dones`：形状 `(time_steps,)` 的布尔数组。
- `gamma`：折扣因子。
- `gae_lambda`：GAE 衰减参数。

## 输出

返回 `(advantages, returns)`，二者均为形状 `(time_steps,)` 的 `float64` 数组，其中 `returns = advantages + values[:-1]`。

## 数学定义

令 $m_t=1-\mathrm{done}_t$，则

$$
\delta_t=r_t+\gamma m_t V_{t+1}-V_t,
$$

$$
A_t=\delta_t+\gamma\lambda m_t A_{t+1}.
$$

## 测试约定

共运行 100 组输入，覆盖单步和长轨迹、中途终止、多段 episode、`gamma` 或 `lambda` 为边界值、零奖励及不同数值尺度。误差标准为 `rtol=1e-11`、`atol=1e-12`。
