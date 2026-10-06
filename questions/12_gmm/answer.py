# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：用 EM 迭代估计全协方差高斯混合模型的权重、均值和协方差。
# API：slogdet 返回行列式符号与对数绝对值；atleast_2d 保证单特征协方差仍是矩阵。

def gmm_em(
    x: np.ndarray,
    n_components: int,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Args:
            x: 形状为 (num_samples, num_features) 的 float64 数组。
            n_components: 混合分量数。
            max_iterations: EM 最大轮数。
            tolerance: 对数似然收敛阈值。
            regularization: 协方差对角线正则量。
            seed: 均值初始化的随机种子。

        Returns:
            返回 `(weights, means, covariances, responsibilities)`，形状分别为 `(K,)`、`(K,D)`、`(K,D,D)`、`(N,K)`。
        """
    # 固定种子的独立生成器负责可复现地抽取初始均值。
    rng = np.random.default_rng(seed)
    # 按数组轴顺序解包 `.shape`，取得后续重排和缓冲区尺寸所需的维度。
    samples, features = x.shape
    # `choice(..., replace=False)` 无放回抽取 K 个不同样本作为初始均值；`.copy()` 避免之后更新均值时改动输入 x。
    means = x[rng.choice(samples, size=n_components, replace=False)].copy()
    # `np.full(K, 1/K)` 让所有高斯分量从均匀先验开始，初始混合权重之和为 1。
    weights = np.full(n_components, 1.0 / n_components)
    # atleast_2d 保证单特征数据的标量协方差也变成 (1, 1) 矩阵。
    base_covariance = np.atleast_2d(np.cov(x, rowvar=False, bias=True))
    # 先给协方差增加对角正则，再用 `np.repeat(..., axis=0)` 沿分量轴复制 K 份，形状变为 (K,D,D)。
    covariances = np.repeat((base_covariance + regularization * np.eye(features))[None, :, :], n_components, axis=0)
    # 保存上一轮总对数似然，用于检测 EM 是否收敛。
    previous_likelihood = -np.inf
    # 最多执行 `max_iterations` 轮迭代；循环内部满足收敛条件时会提前 `break`。
    for _ in range(max_iterations):
        # E 步：计算每个样本在每个高斯分量下的对数联合概率。
        log_probabilities = np.empty((samples, n_components))
        # 按分量编号逐个计算或更新每个高斯分量的参数。
        for component in range(n_components):
            # 计算样本到当前分量均值的偏移，供马氏距离和协方差更新复用。
            difference = x - means[component]
            # slogdet 直接返回对数行列式，避免先算巨大或极小的 det。
            sign, log_determinant = np.linalg.slogdet(covariances[component])
            # 求当前协方差矩阵的逆，用于计算高斯分布的马氏距离。 `np.linalg.inv` 计算方阵的逆。
            inverse = np.linalg.inv(covariances[component])
            # 先算 `(x-μ) @ Σ^-1`，再与 `(x-μ)` 逐元素相乘并沿特征轴求和，得到每个样本的马氏距离平方。
            quadratic = np.sum((difference @ inverse) * difference, axis=1)
            # 写入当前高斯分量的对数先验加对数密度，形成样本-分量联合对数概率。
            log_probabilities[:, component] = np.log(weights[component]) - 0.5 * (features * np.log(2.0 * np.pi) + log_determinant + quadratic)
        # 减去行最大值实现稳定的 log-sum-exp，再归一化为责任度。
        maximum = np.max(log_probabilities, axis=1, keepdims=True)
        # 用 log-sum-exp 合并所有分量的对数概率，得到每个样本的归一化常数。 `np.exp` 对数组逐元素取自然指数；`np.sum` 沿指定轴累加；`keepdims=True` 时保留长度为 1 的归约轴。
        log_normalizer = maximum + np.log(np.sum(np.exp(log_probabilities - maximum), axis=1, keepdims=True))
        # 把对数联合概率减归一化常数后取指数，得到后验责任度。 `np.exp` 对数组逐元素取自然指数。
        responsibilities = np.exp(log_probabilities - log_normalizer)
        # `np.sum` 累加所有样本的 log-normalizer 得到总对数似然；`float` 将 NumPy 标量转为 Python 标量。
        likelihood = float(np.sum(log_normalizer))
        # M 步：责任度之和是各分量的有效样本数，微量常数防止除零。
        effective = np.sum(responsibilities, axis=0) + 1e-15
        # 用各分量有效样本数除以总样本数，更新和为 1 的混合先验。
        weights = effective / samples
        # `responsibilities.T @ x` 对样本作责任度加权求和，再除以有效样本数，得到各分量的新均值。
        means = (responsibilities.T @ x) / effective[:, None]
        # 按分量编号逐个计算或更新每个高斯分量的参数。
        for component in range(n_components):
            # 计算样本到当前分量均值的偏移，供马氏距离和协方差更新复用。
            difference = x - means[component]
            # 用责任度作样本权重，计算当前分量的加权全协方差矩阵。
            covariances[component] = (difference.T * responsibilities[:, component]) @ difference / effective[component]
            # 对角正则保证协方差可逆，避免分量塌缩成奇异矩阵。
            covariances[component] += regularization * np.eye(features)
        # 总对数似然变化足够小时，EM 视为收敛。
        if abs(likelihood - previous_likelihood) <= tolerance:
            # 收敛或终止条件已经满足，提前退出迭代以避免无效计算。
            break
        # 保存上一轮总对数似然，用于检测 EM 是否收敛。
        previous_likelihood = likelihood
    # 返回 EM 最终的混合权重、均值、协方差及最后一轮责任度。
    return weights, means, covariances, responsibilities
