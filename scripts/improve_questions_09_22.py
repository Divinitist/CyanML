from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent, indent


ROOT = Path(__file__).resolve().parents[1]


def item(name: str, description: str) -> tuple[str, str]:
    return name, description


SPECS = [
    dict(id="09", folder="knn", title="KNN", function="knn_predict",
         signature='''def knn_predict(\n    train_x: np.ndarray,\n    train_y: np.ndarray,\n    query_x: np.ndarray,\n    k: int,\n    task: str,\n) -> np.ndarray:''',
         background="K 近邻不训练显式参数，而是直接利用训练样本之间的局部距离完成预测。它既能用于分类，也能用于回归，是理解距离度量、邻域决策和惰性学习的经典算法。",
         requirement="使用欧氏距离，为每个查询样本选取距离最小的 k 个训练样本。分类时进行多数投票，票数相同选择数值更小的标签；回归时返回邻居目标值的算术平均。距离相同时必须保持训练样本的原始顺序。",
         inputs=[item("train_x", "形状为 (num_train, num_features) 的 float64 训练特征。"), item("train_y", "形状为 (num_train,) 的整数标签或浮点回归目标。"), item("query_x", "形状为 (num_queries, num_features) 的 float64 查询特征。"), item("k", "邻居数，满足 1 <= k <= num_train。"), item("task", "字符串 classification 或 regression。")],
         output="返回形状为 `(num_queries,)` 的预测数组。分类结果保持标签 dtype，回归结果为 float64。",
         math="欧氏距离为 $d(x,q)=\\sqrt{\\sum_j(x_j-q_j)^2}$。分类取邻居众数，回归取 $\\hat y=\\frac1k\\sum_{i\\in N_k(q)}y_i$。",
         coverage="分类、回归、不同特征维数与 k、单查询和多查询、负值、距离并列及投票并列", rtol="1e-10", atol="1e-12"),
    dict(id="10", folder="pca", title="PCA", function="pca",
         signature='''def pca(\n    x: np.ndarray,\n    n_components: int,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:''',
         background="主成分分析通过正交方向描述数据中方差最大的变化，是常用的线性降维方法。本题使用 SVD，避免显式构造协方差矩阵带来的额外数值误差。",
         requirement="先按特征求均值并中心化，再对中心化矩阵做经济型 SVD。取前 n_components 个右奇异向量作为主成分。为消除 SVD 的符号不唯一性，每个主成分绝对值最大的元素必须为正。",
         inputs=[item("x", "形状为 (num_samples, num_features) 的 float64 数据矩阵，样本数至少为 2。"), item("n_components", "保留的主成分数，不超过样本数和特征数。")],
         output="依次返回 `transformed`、`components`、`mean`、`explained_variance`。形状分别为 `(N,C)`、`(C,D)`、`(D,)` 和 `(C,)`。",
         math="若中心化矩阵 $X_c=U S V^T$，则投影为 $X_cV_C^T$，解释方差为 $S_C^2/(N-1)$。",
         coverage="不同样本数、特征数、目标维数、相关特征、整体平移、不同数值尺度和非方阵", rtol="1e-8", atol="1e-10"),
    dict(id="11", folder="kmeans", title="K-Means", function="kmeans",
         signature='''def kmeans(\n    x: np.ndarray,\n    n_clusters: int,\n    max_iterations: int,\n    tolerance: float,\n    seed: int,\n) -> tuple[np.ndarray, np.ndarray]:''',
         background="K-Means 交替执行样本分配和中心更新，以最小化簇内平方距离。随机初始化会直接影响结果，因此可复现的随机种子和明确的停止条件都是接口的一部分。",
         requirement="用 `np.random.default_rng(seed)` 无放回选择初始中心。每轮将样本分配给平方欧氏距离最小的中心，再用簇内均值更新中心；空簇保留旧中心。所有中心的最大移动距离不超过 tolerance 时停止，最后用最终中心重新计算标签。",
         inputs=[item("x", "形状为 (num_samples, num_features) 的 float64 数组。"), item("n_clusters", "簇数，不超过样本数。"), item("max_iterations", "最大更新轮数。"), item("tolerance", "非负收敛阈值。"), item("seed", "初始化使用的整数随机种子。")],
         output="返回 `(labels, centers)`；标签形状为 `(N,)`，中心形状为 `(K,D)`。中心编号由随机初始化顺序决定，不做额外排序。",
         math="目标函数为 $\\sum_i\\lVert x_i-\\mu_{c_i}\\rVert_2^2$；更新式为 $\\mu_k=\\frac1{|C_k|}\\sum_{i\\in C_k}x_i$。",
         coverage="不同簇数和维度、明显分簇、重叠分簇、空簇路径、不同迭代上限、容差与固定随机种子", rtol="1e-8", atol="1e-10"),
    dict(id="12", folder="gmm", title="高斯混合模型 GMM", function="gmm_em",
         signature='''def gmm_em(\n    x: np.ndarray,\n    n_components: int,\n    max_iterations: int,\n    tolerance: float,\n    regularization: float,\n    seed: int,\n) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:''',
         background="高斯混合模型用多个高斯分量的加权和描述连续数据。EM 算法在隐含分量归属未知时，交替估计责任度和模型参数。",
         requirement="实现全协方差 GMM。初始均值由固定随机种子无放回抽样，初始权重均匀，各分量使用总体协方差。E 步在对数空间归一化责任度；M 步更新权重、均值和协方差，每个协方差加 regularization 乘单位阵。对数似然变化不超过 tolerance 时停止。",
         inputs=[item("x", "形状为 (num_samples, num_features) 的 float64 数组。"), item("n_components", "混合分量数。"), item("max_iterations", "EM 最大轮数。"), item("tolerance", "对数似然收敛阈值。"), item("regularization", "协方差对角线正则量。"), item("seed", "均值初始化的随机种子。")],
         output="返回 `(weights, means, covariances, responsibilities)`，形状分别为 `(K,)`、`(K,D)`、`(K,D,D)`、`(N,K)`。",
         math="责任度满足 $r_{ik}=\\frac{\\pi_k\\mathcal N(x_i|\\mu_k,\\Sigma_k)}{\\sum_j\\pi_j\\mathcal N(x_i|\\mu_j,\\Sigma_j)}$，每行之和为 1。",
         coverage="一至三维、不同分量数、分离和重叠分布、不同正则量、迭代上限与固定种子", rtol="1e-7", atol="1e-9"),
    dict(id="13", folder="decision_tree_split", title="决策树最佳划分", function="best_decision_tree_split",
         signature='''def best_decision_tree_split(\n    x: np.ndarray,\n    labels: np.ndarray,\n    criterion: str,\n) -> tuple[int, float, float]:''',
         background="分类决策树通过选择特征和阈值，使子节点比父节点更纯。本题只实现一个节点的穷举划分，是 CART 和信息增益树的核心步骤。",
         requirement="对每个特征排序后的相邻不同取值取中点作为候选阈值，计算父节点纯度减去左右子节点加权纯度的增益。criterion 支持 gini 和 entropy。增益并列时保留特征下标更小、阈值更小的候选。",
         inputs=[item("x", "形状为 (num_samples, num_features) 的 float64 特征矩阵。"), item("labels", "形状为 (num_samples,) 的整数分类标签。"), item("criterion", "字符串 gini 或 entropy。")],
         output="返回 `(feature_index, threshold, gain)`，分别是最佳特征下标、浮点阈值和纯度增益。输入保证至少存在一个有效候选阈值。",
         math="Gini 为 $1-\\sum_cp_c^2$，Entropy 为 $-\\sum_cp_c\\log_2p_c$；增益为 $I(P)-\\frac{N_L}{N}I(L)-\\frac{N_R}{N}I(R)$。",
         coverage="两种纯度、多分类、重复特征值、完美划分、弱增益、多特征及并列决策", rtol="1e-9", atol="1e-11"),
]


ATTENTION = [
    ("14", "multi_head_attention", "Multi-Head Attention", "multi_head_attention",
     '''def multi_head_attention(\n    x: np.ndarray,\n    w_q: np.ndarray,\n    w_k: np.ndarray,\n    w_v: np.ndarray,\n    w_o: np.ndarray,\n    num_heads: int,\n    mask: np.ndarray | None = None,\n) -> np.ndarray:''',
     "多头注意力让模型在多个独立子空间中建立序列位置之间的联系，再合并各头信息。它是 Transformer 的核心算子，本题实现不含偏置和 dropout 的单批次版本。",
     "分别计算 Q、K、V 投影，将模型维度均分为 num_heads 个头，执行缩放点积注意力后拼接所有头并乘输出矩阵。可选布尔 mask 中 True 表示允许关注；每个查询位置保证至少有一个可见键。",
     [item("x", "形状为 (sequence_length, model_dim) 的 float64 序列。"), item("w_q / w_k / w_v / w_o", "四个形状均为 (model_dim, model_dim) 的投影矩阵。"), item("num_heads", "注意力头数，能整除 model_dim。"), item("mask", "None 或形状为 (L,L) 的布尔数组。")],
     "返回形状为 `(L, model_dim)` 的注意力输出。", "$A=\\operatorname{softmax}(QK^T/\\sqrt{d_h})$，每个头的上下文为 $AV$。", "不同序列长度、头数、头维度、无掩码与因果掩码", "1e-9", "1e-11"),
    ("15", "multi_query_attention", "Multi-Query Attention", "multi_query_attention",
     '''def multi_query_attention(\n    x: np.ndarray,\n    w_q: np.ndarray,\n    w_k: np.ndarray,\n    w_v: np.ndarray,\n    w_o: np.ndarray,\n    num_query_heads: int,\n    mask: np.ndarray | None = None,\n) -> np.ndarray:''',
     "多查询注意力保留多个查询头，但让所有查询头共享同一组键和值。与标准多头注意力相比，它显著减少自回归推理时 KV Cache 的体积。",
     "w_q 投影到完整模型维度，w_k 和 w_v 仅投影到一个头维度。每个查询头分别与共享 K/V 计算缩放点积注意力，随后拼接并做输出投影。布尔掩码语义与多头注意力一致。",
     [item("x", "形状为 (L,D) 的 float64 输入。"), item("w_q", "形状为 (D,D)。"), item("w_k / w_v", "形状为 (D, head_dim)。"), item("w_o", "形状为 (D,D)。"), item("num_query_heads", "查询头数，head_dim=D/H。"), item("mask", "None 或形状为 (L,L) 的布尔掩码。")],
     "返回形状为 `(L,D)` 的 float64 数组。", "$A_h=\\operatorname{softmax}(Q_hK^T/\\sqrt{d_h})$；所有 h 使用相同的 K 和 V。", "不同查询头数、头维度、序列长度、共享 K/V、无掩码和因果掩码", "1e-8", "1e-10"),
    ("16", "grouped_query_attention", "Grouped-Query Attention", "grouped_query_attention",
     '''def grouped_query_attention(\n    x: np.ndarray,\n    w_q: np.ndarray,\n    w_k: np.ndarray,\n    w_v: np.ndarray,\n    w_o: np.ndarray,\n    num_query_heads: int,\n    num_kv_heads: int,\n    mask: np.ndarray | None = None,\n) -> np.ndarray:''',
     "分组查询注意力位于 MHA 与 MQA 之间：查询头分成若干组，每组共享一个 K/V 头。它在模型质量、KV 缓存大小和推理吞吐之间提供折中。",
     "查询头数必须能被 KV 头数整除。将连续的 num_query_heads/num_kv_heads 个查询头映射到同一个 KV 头，计算缩放点积注意力，拼接后通过输出投影。",
     [item("x", "形状为 (L,D) 的 float64 输入。"), item("w_q", "形状为 (D,D) 的查询投影。"), item("w_k / w_v", "形状为 (D, num_kv_heads*head_dim)。"), item("w_o", "形状为 (D,D)。"), item("num_query_heads / num_kv_heads", "查询头数和 KV 头数。"), item("mask", "None 或 (L,L) 布尔掩码。")],
     "返回形状为 `(L,D)` 的注意力输出。", "令 $g=H_q/H_{kv}$，查询头 h 使用 KV 头 $\\lfloor h/g\\rfloor$。注意力缩放因子仍为 $\\sqrt{d_h}$。", "MHA 退化情形、MQA 退化情形、多个分组、不同掩码和序列长度", "1e-8", "1e-10"),
    ("17", "cross_attention", "Cross Attention", "cross_attention",
     '''def cross_attention(\n    query: np.ndarray,\n    context: np.ndarray,\n    w_q: np.ndarray,\n    w_k: np.ndarray,\n    w_v: np.ndarray,\n    w_o: np.ndarray,\n    num_heads: int,\n    mask: np.ndarray | None = None,\n) -> np.ndarray:''',
     "交叉注意力让一个序列从另一个序列读取信息，常用于编码器—解码器模型和多模态模型。查询长度与上下文长度可以不同。",
     "Q 只由 query 投影，K/V 只由 context 投影。按 num_heads 分头计算缩放点积注意力，再拼接并输出投影。mask 的第一维对应查询位置，第二维对应上下文位置，True 表示允许读取。",
     [item("query", "形状为 (query_length, model_dim)。"), item("context", "形状为 (context_length, model_dim)。"), item("w_q / w_k / w_v / w_o", "形状均为 (model_dim, model_dim)。"), item("num_heads", "能整除模型维度的头数。"), item("mask", "None 或形状为 (Q,K) 的布尔掩码。")],
     "返回形状为 `(query_length, model_dim)` 的 float64 数组。", "$A=\\operatorname{softmax}(Q_{query}K_{context}^T/\\sqrt{d_h})$，输出为 $AV_{context}$。", "不等长 Q/KV、不同头数、随机稀疏掩码、无掩码、单查询位置", "1e-8", "1e-10"),
    ("18", "sliding_window_attention", "Sliding-Window Attention", "sliding_window_attention",
     '''def sliding_window_attention(\n    x: np.ndarray,\n    w_q: np.ndarray,\n    w_k: np.ndarray,\n    w_v: np.ndarray,\n    w_o: np.ndarray,\n    num_heads: int,\n    window_size: int,\n) -> np.ndarray:''',
     "滑动窗口注意力把全局二次复杂度限制为局部窗口，适合处理长序列。本题使用因果窗口，因此当前位置不能读取未来 token。",
     "完成标准多头 Q/K/V 投影。位置 i 只能关注从 max(0,i-window_size+1) 到 i 的键；在允许区域内计算稳定 softmax，拼接各头并做输出投影。window_size 大于序列长度时等价于完整因果注意力。",
     [item("x", "形状为 (L,D) 的 float64 序列。"), item("w_q / w_k / w_v / w_o", "四个 (D,D) 投影矩阵。"), item("num_heads", "能整除 D 的注意力头数。"), item("window_size", "正整数窗口宽度，包含当前位置。")],
     "返回形状为 `(L,D)` 的局部因果注意力结果。", "可见条件为 $j\\le i$ 且 $j\\ge i-w+1$；可见分数再按最后一维执行 softmax。", "窗口为 1、窗口覆盖全序列、不同长度、不同头数和边界位置", "1e-8", "1e-10"),
]


POSITION = [
    ("19", "absolute_position_encoding", "绝对位置编码", "add_sinusoidal_position_encoding",
     '''def add_sinusoidal_position_encoding(\n    x: np.ndarray,\n    start_position: int = 0,\n    base: float = 10000.0,\n) -> np.ndarray:''',
     "不含循环和卷积的注意力本身无法区分 token 顺序。正弦绝对位置编码使用不同频率的 sin/cos 信号，把确定性的位置信息直接加到词向量上。",
     "为 x 的每个位置构造位置编码。偶数维使用 sin，奇数维使用 cos；奇数模型维度时最后一个偶数维仍需正确计算。start_position 允许为分块序列指定全局起始位置，函数不得修改原数组。",
     [item("x", "形状为 (sequence_length, model_dim) 的 float64 数组。"), item("start_position", "第一个 token 的非负绝对位置。"), item("base", "控制频率跨度的正浮点数。")],
     "返回与 x 同形状的 `x + position_encoding`。", "$PE_{p,2i}=\\sin(p/base^{2i/d})$，$PE_{p,2i+1}=\\cos(p/base^{2i/d})$。", "奇偶模型维度、不同起始位置、序列长度、base 和输入数值", "1e-10", "1e-12"),
    ("20", "rope", "RoPE", "apply_rope",
     '''def apply_rope(\n    q: np.ndarray,\n    k: np.ndarray,\n    position_ids: np.ndarray,\n    base: float = 10000.0,\n    rotary_dim: int | None = None,\n) -> tuple[np.ndarray, np.ndarray]:''',
     "旋转位置编码把绝对位置表示为 Q/K 各二维子空间中的旋转，使点积自然依赖相对位置。它不向残差流直接添加位置向量，是现代解码器模型的常见选择。",
     "对 Q/K 的前 rotary_dim 维按相邻偶、奇维配对旋转，其余维保持不变。rotary_dim=None 表示旋转整个头维度。输入旋转维数保证为正偶数，position_ids 可从任意非负位置开始。",
     [item("q / k", "形状为 (sequence_length, num_heads, head_dim) 的 float64 数组。"), item("position_ids", "形状为 (sequence_length,) 的位置数组。"), item("base", "RoPE 频率基数。"), item("rotary_dim", "None 或不超过 head_dim 的正偶数。")],
     "返回旋转后的 `(q_rotated, k_rotated)`，两者形状不变且不修改输入。", "对每对 $(x_0,x_1)$ 应用 $[x_0\\cos\\theta-x_1\\sin\\theta, x_0\\sin\\theta+x_1\\cos\\theta]$。", "部分和全部旋转维、不同头数、位置偏移、头维度和保持不变的尾部", "1e-8", "1e-10"),
    ("21", "long_context_rope", "长上下文 RoPE", "apply_scaled_rope",
     '''def apply_scaled_rope(\n    q: np.ndarray,\n    k: np.ndarray,\n    position_ids: np.ndarray,\n    scale: float,\n    method: str,\n    base: float = 10000.0,\n) -> tuple[np.ndarray, np.ndarray]:''',
     "直接把 RoPE 外推到远超训练长度的位置可能导致频率分布失配。线性插值和动态 NTK 基数缩放是两种常见的简化扩展策略。",
     "method=linear 时将所有 position_ids 除以 scale 后使用原 base；method=ntk 时位置不变，将基数改为 base*scale^(d/(d-2))。随后对全部偶数头维执行标准 RoPE 旋转，头维保证为大于 2 的偶数。",
     [item("q / k", "形状为 (L,H,d) 的 float64 数组。"), item("position_ids", "形状为 (L,) 的绝对位置。"), item("scale", "大于等于 1 的上下文扩展倍率。"), item("method", "字符串 linear 或 ntk。"), item("base", "原始 RoPE 基数。")],
     "返回缩放旋转后的 Q/K 二元组，形状与输入一致。", "linear 使用 $p'=p/s$；ntk 使用 $base'=base\\cdot s^{d/(d-2)}$，频率仍为 $base'^{-2i/d}$。", "两种扩展方法、不同倍率、大位置编号、多个头和多个偶数头维", "1e-8", "1e-10"),
    ("22", "relative_position_encoding", "相对位置编码", "relative_position_bias",
     '''def relative_position_bias(\n    query_length: int,\n    key_length: int,\n    num_heads: int,\n    mode: str,\n    max_distance: int,\n    table: np.ndarray | None = None,\n) -> np.ndarray:''',
     "相对位置偏置直接修改注意力 logits，让模型根据查询和键之间的距离调整注意力。它可以是固定函数，也可以来自可学习查找表。",
     "支持 alibi 和 table 两种模式。alibi 为每个头使用斜率 2^(-8(h+1)/H)，偏置为斜率乘负绝对距离；table 模式把有符号距离 key-query 截断到给定范围，再查询每个头自己的表。",
     [item("query_length / key_length", "查询和键的正整数长度，可以不同。"), item("num_heads", "注意力头数。"), item("mode", "字符串 alibi 或 table。"), item("max_distance", "table 模式的最大有符号距离。"), item("table", "table 模式下形状为 (H,2*max_distance+1)；alibi 时为 None。")],
     "返回形状为 `(num_heads, query_length, key_length)` 的 float64 偏置，可直接加到注意力分数。", "相对距离 $r=j-i$；table 索引为 $clip(r,-M,M)+M$，ALiBi 为 $-2^{-8(h+1)/H}|r|$。", "两种模式、不等长序列、不同头数、距离截断、单位置及多种表值", "1e-8", "1e-10"),
]


for values in ATTENTION + POSITION:
    identifier, folder, title, function, signature, background, requirement, inputs, output, math, coverage, rtol, atol = values
    SPECS.append(dict(id=identifier, folder=folder, title=title, function=function, signature=signature,
                      background=background, requirement=requirement, inputs=inputs, output=output,
                      math=math, coverage=coverage, rtol=rtol, atol=atol))


def rewrite(spec: dict) -> None:
    folder = ROOT / "questions" / f"{spec['id']}_{spec['folder']}"
    input_lines = "\n".join(f"- `{name}`：{description}" for name, description in spec["inputs"])
    problem = f'''# {spec["title"]}

## 背景

{spec["background"]} 本题要求只使用 NumPy 完成核心计算，以便直接观察张量形状和数值过程。

## 需求

实现 `{spec["function"]}`。{spec["requirement"]} 所有输入均保证合法，不需要处理题目约定以外的类型、形状或取值。

## 输入

{input_lines}

除特别说明外，参与数值计算的数组和常量统一按 `float64` 处理。函数不应修改调用者传入的数组。

## 输出

{spec["output"]}

## 数学定义

{spec["math"]}

## 测试约定

固定随机种子生成 100 组输入输出，覆盖{spec["coverage"]}。本题判定误差为 `rtol={spec["rtol"]}`、`atol={spec["atol"]}`。测试只调用公开函数，不依赖标准输出；提交代码中可以定义私有辅助函数。
'''
    (folder / "problem.md").write_text(problem, encoding="utf-8")

    args_doc = "\n".join(f"        {name}: {description}" for name, description in spec["inputs"])
    doc = f'''"""
    Args:
{args_doc}

    Returns:
        {spec["output"]}
    """'''
    signature = dedent(spec["signature"]).strip()
    base = "import numpy as np\n\n\n" + signature + "\n" + indent(dedent(doc), "    ") + "\n"

    old_answer = (folder / "answer.py").read_text(encoding="utf-8")
    tree = ast.parse(old_answer)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    first_code = function.body[1] if isinstance(function.body[0], ast.Expr) else function.body[0]
    body = "\n".join(old_answer.splitlines()[first_code.lineno - 1:]).rstrip() + "\n"
    (folder / "answer.py").write_text(base + body, encoding="utf-8")
    starter = base + "    pass\n"
    (folder / "template.py").write_text(starter, encoding="utf-8")
    (folder / "submission.py").write_text(starter, encoding="utf-8")


for question in SPECS:
    rewrite(question)
