from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent


ROOT = Path(__file__).resolve().parents[1]


def write_question(spec: dict[str, str]) -> None:
    folder = ROOT / "questions" / f"{spec['id']}_{spec['slug']}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": spec["id"],
        "slug": spec["slug"].replace("_", "-"),
        "title": spec["title"],
        "entrypoint": spec["function"],
        "library": "NumPy",
        "difficulty": spec.get("difficulty", "中等"),
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (folder / "problem.md").write_text(dedent(spec["problem"]).strip() + "\n", encoding="utf-8")
    base = "import numpy as np\n\n\n" + dedent(spec["signature"]).strip() + "\n"
    docstring = indent(dedent(spec["docstring"]).strip(), "    ")
    body = indent(dedent(spec["answer"]).strip(), "    ")
    template = base + docstring + "\n    pass\n"
    (folder / "template.py").write_text(template, encoding="utf-8")
    (folder / "submission.py").write_text(template, encoding="utf-8")
    answer = base + docstring + "\n" + body + "\n"
    (folder / "answer.py").write_text(answer, encoding="utf-8")
    test = dedent(
        f'''\
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path

        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from test_support import run_cases

        from answer import {spec["function"]} as reference
        from submission import {spec["function"]}


        {indent(dedent(spec["cases"]).strip(), "        ").lstrip()}


        def run():
            return run_cases(
                build_cases(),
                {spec["function"]},
                reference,
                rtol={spec.get("rtol", "1e-8")},
                atol={spec.get("atol", "1e-10")},
                mismatch_message="输出与参考实现不一致。",
            )


        if __name__ == "__main__":
            print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
        '''
    )
    (folder / "test.py").write_text(test, encoding="utf-8")


COMMON_DOC = '''\
    """Return the values described in problem.md for valid float64 inputs."""
'''


SPECS = [
    {
        "id": "09", "slug": "knn", "title": "KNN", "function": "knn_predict",
        "signature": '''\
def knn_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    k: int,
    task: str,
) -> np.ndarray:''',
        "docstring": COMMON_DOC,
        "problem": '''\
# KNN

## 需求

实现 `knn_predict(train_x, train_y, query_x, k, task)`。使用欧氏距离寻找每个查询样本最近的 `k` 个训练样本。`task="classification"` 时进行多数投票，票数相同选择数值更小的标签；`task="regression"` 时返回邻居目标的算术平均值。

输入矩阵均为 `float64`，分类标签为整数。返回形状 `(num_queries,)`。100 组测试覆盖分类、回归、距离并列、不同维数与 k 值；误差为 `rtol=1e-10`、`atol=1e-12`。
''',
        "answer": '''\
    distances = np.sqrt(np.sum((query_x[:, None, :] - train_x[None, :, :]) ** 2, axis=2))
    neighbor_indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    neighbor_targets = train_y[neighbor_indices]
    if task == "regression":
        return np.mean(neighbor_targets, axis=1)
    predictions = []
    for targets in neighbor_targets:
        labels, counts = np.unique(targets, return_counts=True)
        predictions.append(labels[np.argmax(counts)])
    return np.asarray(predictions, dtype=train_y.dtype)
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        samples = 5 + index % 16
        features = 1 + index % 6
        queries = 1 + index % 7
        train_x = rng.normal(size=(samples, features)).astype(np.float64)
        query_x = rng.normal(size=(queries, features)).astype(np.float64)
        task = "classification" if index % 2 == 0 else "regression"
        train_y = rng.integers(0, 4, size=samples) if task == "classification" else rng.normal(size=samples)
        k = 1 + index % min(7, samples)
        cases.append({"name": f"{task} k={k}", "args": (train_x, train_y, query_x, k, task)})
    return cases
''', "rtol": "1e-10", "atol": "1e-12",
    },
    {
        "id": "10", "slug": "pca", "title": "PCA", "function": "pca",
        "signature": '''\
def pca(
    x: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:''',
        "docstring": COMMON_DOC,
        "problem": '''\
# PCA

## 需求

实现 `pca(x, n_components)`：中心化数据，通过 SVD 提取主成分，并返回 `(transformed, components, mean, explained_variance)`。`components` 形状为 `(n_components, num_features)`；为消除符号不唯一性，每个主成分绝对值最大的元素必须为正。解释方差使用 `S²/(N-1)`。

100 组测试覆盖降到不同维数、相关特征、平移和不同数据尺度。默认 `float64`，误差为 `rtol=1e-8`、`atol=1e-10`。
''',
        "answer": '''\
    mean = np.mean(x, axis=0)
    centered = x - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:n_components].copy()
    for row in range(components.shape[0]):
        pivot = np.argmax(np.abs(components[row]))
        if components[row, pivot] < 0:
            components[row] *= -1.0
    transformed = centered @ components.T
    explained_variance = singular_values[:n_components] ** 2 / (x.shape[0] - 1)
    return transformed, components, mean, explained_variance
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        samples = 5 + index % 15
        features = 2 + index % 7
        latent = rng.normal(size=(samples, features))
        mixing = rng.normal(size=(features, features))
        x = (latent @ mixing + rng.normal(size=features) * (index % 5)).astype(np.float64)
        n_components = 1 + index % min(samples - 1, features)
        cases.append({"name": f"{features}维降至{n_components}维", "args": (x, n_components)})
    return cases
''',
    },
    {
        "id": "11", "slug": "kmeans", "title": "K-Means", "function": "kmeans",
        "signature": '''\
def kmeans(
    x: np.ndarray,
    n_clusters: int,
    max_iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:''',
        "docstring": COMMON_DOC,
        "problem": '''\
# K-Means

## 需求

实现 `kmeans(x, n_clusters, max_iterations, tolerance, seed)`。使用给定随机种子无放回选择初始中心，反复执行最近中心分配和均值更新；空簇保留旧中心。当所有中心的最大移动距离不超过 `tolerance` 时停止。

返回 `(labels, centers)`。100 组测试使用固定种子，覆盖不同簇数、维数、明显分簇和可能出现空簇的输入。
''',
        "answer": '''\
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(x.shape[0], size=n_clusters, replace=False)].copy()
    for _ in range(max_iterations):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster in range(n_clusters):
            members = x[labels == cluster]
            if members.size:
                new_centers[cluster] = np.mean(members, axis=0)
        movement = np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))
        centers = new_centers
        if np.max(movement) <= tolerance:
            break
    labels = np.argmin(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
    return labels, centers
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        clusters = 2 + index % 4
        features = 1 + index % 4
        points_per = 4 + index % 6
        centers = rng.normal(scale=4.0, size=(clusters, features))
        x = np.concatenate([center + rng.normal(scale=0.4, size=(points_per, features)) for center in centers]).astype(np.float64)
        cases.append({"name": f"{clusters}簇 {features}维", "args": (x, clusters, 30, 1e-8, 1000 + index)})
    return cases
''',
    },
    {
        "id": "12", "slug": "gmm", "title": "GMM", "function": "gmm_em",
        "signature": '''\
def gmm_em(
    x: np.ndarray,
    n_components: int,
    max_iterations: int,
    tolerance: float,
    regularization: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:''',
        "docstring": COMMON_DOC,
        "problem": '''\
# 高斯混合模型 GMM

## 需求

实现全协方差 GMM 的 EM 算法 `gmm_em`。用给定随机种子无放回选择初始均值，初始混合权重均匀、各分量协方差使用总体协方差加 `regularization·I`。E-step 计算责任度，M-step 更新权重、均值和协方差；对数似然变化不超过 `tolerance` 时停止。

返回 `(weights, means, covariances, responsibilities)`。100 组小规模测试覆盖不同分量数和维数，误差为 `rtol=1e-7`、`atol=1e-9`。
''',
        "answer": '''\
    rng = np.random.default_rng(seed)
    samples, features = x.shape
    means = x[rng.choice(samples, size=n_components, replace=False)].copy()
    weights = np.full(n_components, 1.0 / n_components)
    base_covariance = np.atleast_2d(np.cov(x, rowvar=False, bias=True))
    covariances = np.repeat((base_covariance + regularization * np.eye(features))[None, :, :], n_components, axis=0)
    previous_likelihood = -np.inf
    for _ in range(max_iterations):
        log_probabilities = np.empty((samples, n_components))
        for component in range(n_components):
            difference = x - means[component]
            sign, log_determinant = np.linalg.slogdet(covariances[component])
            inverse = np.linalg.inv(covariances[component])
            quadratic = np.sum((difference @ inverse) * difference, axis=1)
            log_probabilities[:, component] = np.log(weights[component]) - 0.5 * (features * np.log(2.0 * np.pi) + log_determinant + quadratic)
        maximum = np.max(log_probabilities, axis=1, keepdims=True)
        log_normalizer = maximum + np.log(np.sum(np.exp(log_probabilities - maximum), axis=1, keepdims=True))
        responsibilities = np.exp(log_probabilities - log_normalizer)
        likelihood = float(np.sum(log_normalizer))
        effective = np.sum(responsibilities, axis=0) + 1e-15
        weights = effective / samples
        means = (responsibilities.T @ x) / effective[:, None]
        for component in range(n_components):
            difference = x - means[component]
            covariances[component] = (difference.T * responsibilities[:, component]) @ difference / effective[component]
            covariances[component] += regularization * np.eye(features)
        if abs(likelihood - previous_likelihood) <= tolerance:
            break
        previous_likelihood = likelihood
    return weights, means, covariances, responsibilities
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        components = 2 + index % 3
        features = 1 + index % 3
        points = 12 + index % 8
        anchors = rng.normal(scale=3.0, size=(components, features))
        x = np.vstack([anchors[i % components] + rng.normal(scale=0.5, size=features) for i in range(points)]).astype(np.float64)
        cases.append({"name": f"{components}分量 {features}维", "args": (x, components, 12, 1e-7, 1e-5, 2000 + index)})
    return cases
''', "rtol": "1e-7", "atol": "1e-9",
    },
    {
        "id": "13", "slug": "decision_tree_split", "title": "决策树划分", "function": "best_decision_tree_split",
        "signature": '''\
def best_decision_tree_split(
    x: np.ndarray,
    labels: np.ndarray,
    criterion: str,
) -> tuple[int, float, float]:''',
        "docstring": COMMON_DOC,
        "problem": '''\
# 决策树最佳划分

## 需求

实现 `best_decision_tree_split(x, labels, criterion)`。枚举每个特征相邻不同取值的中点作为阈值，计算父节点纯度减去左右子节点加权纯度的增益。`criterion` 为 `"gini"` 或 `"entropy"`。并列时保留特征下标更小、阈值更小的候选。

返回 `(feature_index, threshold, gain)`。100 组测试覆盖两种纯度、多分类、重复特征值和完美划分。
''',
        "answer": '''\
    def impurity(values):
        if values.size == 0:
            return 0.0
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / values.size
        if criterion == "gini":
            return float(1.0 - np.sum(probabilities ** 2))
        return float(-np.sum(probabilities * np.log2(probabilities)))

    parent = impurity(labels)
    best_feature, best_threshold, best_gain = 0, 0.0, -np.inf
    for feature in range(x.shape[1]):
        values = np.unique(x[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left = x[:, feature] <= threshold
            right = ~left
            weighted = left.mean() * impurity(labels[left]) + right.mean() * impurity(labels[right])
            gain = parent - weighted
            if gain > best_gain + 1e-15:
                best_feature, best_threshold, best_gain = feature, float(threshold), float(gain)
    return best_feature, best_threshold, best_gain
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        samples = 8 + index % 15
        features = 1 + index % 6
        classes = 2 + index % 3
        x = rng.integers(-4, 7, size=(samples, features)).astype(np.float64)
        labels = ((x[:, index % features] > (index % 3 - 1)).astype(int) + rng.integers(0, classes, size=samples) * (index % 4 == 0)) % classes
        criterion = "gini" if index % 2 == 0 else "entropy"
        cases.append({"name": f"{criterion} {classes}类", "args": (x, labels, criterion)})
    return cases
''',
    },
]


for question in SPECS:
    write_question(question)
