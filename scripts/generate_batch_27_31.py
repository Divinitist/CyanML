from rich_question_codegen import rich_generate_all


SPECS = [
    dict(
        id="27", slug="kv_cache", title="通用 KV Cache", function="kv_cache_attention",
        signature='''\
def kv_cache_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    cache_k: np.ndarray | None = None,
    cache_v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:''',
        background="自回归推理若每一步都重新投影整个前缀，会重复计算历史 token 的 K/V。KV Cache 保存这些投影，使预填充和逐 token 解码共享同一个注意力接口。",
        requirement="实现同时支持 MHA、MQA 和 GQA 的分块因果注意力。只把当前 x 投影出的 K/V 追加到缓存；查询头按连续分组复用 KV 头。当前分块第 i 个查询能看到全部历史缓存及当前块中不晚于 i 的键。cache 为 None 时视为空缓存，返回更新后的缓存。",
        inputs=[("x", "当前分块，形状为 (chunk_length, model_dim)。"), ("w_q", "形状为 (D,D) 的查询投影。"), ("w_k / w_v", "形状为 (D,num_kv_heads*head_dim) 的 KV 投影。"), ("w_o", "形状为 (D,D) 的输出投影。"), ("num_query_heads / num_kv_heads", "查询头数与 KV 头数，前者可被后者整除。"), ("cache_k / cache_v", "None 或形状为 (num_kv_heads,past_length,head_dim) 的历史缓存。")],
        output="返回 `(output, new_cache_k, new_cache_v)`；output 为 `(chunk_length,D)`，新缓存长度为 past_length+chunk_length。",
        math="查询头 h 使用 KV 头 $\\lfloor h/(H_q/H_{kv})\\rfloor$。当前块查询 i 的可见键满足 $j\\le past+i$，注意力仍按 $1/\\sqrt{d_h}$ 缩放。",
        coverage="空缓存预填充、非空缓存解码、单 token 与多 token 块、MHA/MQA/GQA、不同历史长度和头维", rtol="1e-8", atol="1e-10",
        answer='''\
chunk_length, model_dim = x.shape
head_dim = model_dim // num_query_heads
q = (x @ w_q).reshape(chunk_length, num_query_heads, head_dim).transpose(1, 0, 2)
new_k = (x @ w_k).reshape(chunk_length, num_kv_heads, head_dim).transpose(1, 0, 2)
new_v = (x @ w_v).reshape(chunk_length, num_kv_heads, head_dim).transpose(1, 0, 2)
if cache_k is None:
    all_k, all_v, past = new_k, new_v, 0
else:
    past = cache_k.shape[1]
    all_k = np.concatenate((cache_k, new_k), axis=1)
    all_v = np.concatenate((cache_v, new_v), axis=1)
repeat = num_query_heads // num_kv_heads
expanded_k = np.repeat(all_k, repeat, axis=0)
expanded_v = np.repeat(all_v, repeat, axis=0)
scores = q @ expanded_k.transpose(0, 2, 1) / np.sqrt(head_dim)
mask = np.arange(all_k.shape[1])[None, :] <= past + np.arange(chunk_length)[:, None]
scores = np.where(mask[None], scores, -np.inf)
weights = np.exp(scores - scores.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
output = (weights @ expanded_v).transpose(1, 0, 2).reshape(chunk_length, model_dim) @ w_o
return output, all_k, all_v
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(270027); cases = []
    for index in range(100):
        q_heads = (1, 2, 4, 8)[index % 4]; kv_heads = (1, q_heads)[index % 2]
        head_dim, chunk, past = 2 + index % 3, 1 + index % 4, index % 6
        dim = q_heads * head_dim; x = rng.normal(size=(chunk, dim))
        w_q = rng.normal(scale=.25, size=(dim, dim)); w_k = rng.normal(scale=.25, size=(dim, kv_heads * head_dim)); w_v = rng.normal(scale=.25, size=(dim, kv_heads * head_dim)); w_o = rng.normal(scale=.25, size=(dim, dim))
        cache_k = None if past == 0 else rng.normal(size=(kv_heads, past, head_dim)); cache_v = None if past == 0 else rng.normal(size=(kv_heads, past, head_dim))
        cases.append({"name": f"q={q_heads}, kv={kv_heads}, past={past}, chunk={chunk}", "args": (x, w_q, w_k, w_v, w_o, q_heads, kv_heads, cache_k, cache_v)})
    return cases
'''),
    dict(
        id="28", slug="llm_sampling", title="LLM 采样", function="sample_next_token",
        signature='''\
def sample_next_token(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> tuple[int, np.ndarray]:''',
        background="语言模型把下一 token 表示为词表上的 logits。温度、Top-K 和 Top-P 依次改变候选分布，在确定性与多样性之间进行控制；固定随机种子使采样可复现。",
        requirement="先将 logits 除以 temperature，再保留最大的 top_k 项；top_k=0 表示不过滤。随后在当前候选上按概率降序保留累计概率首次达到 top_p 所需的最小集合，边界 token 必须保留。把被过滤项概率置零并重新归一化，最后使用 default_rng(seed).choice 采样。",
        inputs=[("logits", "形状为 (vocab_size,) 的 float64 未归一化分数。"), ("temperature", "大于 0 的温度。"), ("top_k", "0 或不超过词表大小的候选数。"), ("top_p", "位于 (0,1] 的核采样阈值。"), ("seed", "单次采样的整数随机种子。")],
        output="返回 `(token_id, probabilities)`；token_id 为 Python int，probabilities 形状为 `(vocab_size,)`、总和为 1。",
        math="$p_i=softmax(z_i/T)$。Top-P 选择按概率降序的最短前缀 S，使 $\\sum_{i\\in S}p_i\\ge p$，过滤后再次归一化。",
        coverage="不同词表、温度、Top-K 开关、Top-P 边界、极端 logits、概率并列和固定种子复现", rtol="1e-10", atol="1e-12",
        answer='''\
scaled = logits / temperature
keep = np.ones(logits.size, dtype=bool)
if top_k > 0 and top_k < logits.size:
    order = np.argsort(-scaled, kind="stable")
    keep[:] = False; keep[order[:top_k]] = True
filtered = np.where(keep, scaled, -np.inf)
probabilities = np.exp(filtered - np.max(filtered)); probabilities /= probabilities.sum()
if top_p < 1.0:
    order = np.argsort(-probabilities, kind="stable")
    cumulative = np.cumsum(probabilities[order])
    count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
    nucleus = np.zeros_like(keep); nucleus[order[:count]] = True
    probabilities = np.where(nucleus, probabilities, 0.0); probabilities /= probabilities.sum()
token = int(np.random.default_rng(seed).choice(logits.size, p=probabilities))
return token, probabilities
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(280028); cases = []
    for index in range(100):
        vocab = 2 + index % 19; logits = rng.normal(scale=1 + index % 4, size=vocab)
        temperature = (.5, .8, 1., 1.5, 2.)[index % 5]; top_k = (0, 1, min(3, vocab), min(7, vocab))[index % 4]; top_p = (.35, .7, .9, 1.)[index % 4]
        cases.append({"name": f"vocab={vocab}, k={top_k}, p={top_p}", "args": (logits, temperature, top_k, top_p, 9000 + index)})
    return cases
'''),
    dict(
        id="29", slug="beam_search", title="Beam Search", function="beam_search",
        signature='''\
def beam_search(
    transition_log_probs: np.ndarray,
    start_token: int,
    end_token: int,
    max_new_tokens: int,
    beam_width: int,
    length_penalty: float,
) -> tuple[list[int], float]:''',
        background="Beam Search 在每一步保留若干累计分数最高的序列，比贪心搜索探索更多路径。本题用固定的一阶转移表代替神经网络，以隔离搜索算法本身。",
        requirement="序列从 start_token 开始，每一步用最后一个 token 对应的行取得下一 token 对数概率。已生成 end_token 的 beam 不再扩展。候选按长度惩罚后的分数降序保留 beam_width 个；完全并列时按 token 序列字典序升序。达到最大步数或所有 beam 完成后，返回同一规则下最优项。",
        inputs=[("transition_log_probs", "形状为 (vocab_size,vocab_size) 的一阶转移对数概率。"), ("start_token / end_token", "合法的起始和终止 token 下标。"), ("max_new_tokens", "最多追加的 token 数。"), ("beam_width", "每轮保留的 beam 数。"), ("length_penalty", "非负长度惩罚指数。")],
        output="返回 `(tokens, score)`；tokens 包含 start_token，score 是最终累计对数概率除以生成长度的 length_penalty 次方。",
        math="若生成长度（不含起始 token）为 L，排序分数为 $s(y)=\\sum_t\\log p(y_t|y_{t-1})/L^\\alpha$；未生成 token 时分母按 1 处理。",
        coverage="贪心退化、多个 beam、提前结束、无结束路径、长度惩罚、分数并列和不同词表大小", rtol="1e-10", atol="1e-12",
        answer='''\
def normalized(tokens, total):
    length = max(1, len(tokens) - 1)
    return total / (length ** length_penalty)

beams = [([int(start_token)], 0.0)]
for _ in range(max_new_tokens):
    candidates = []
    for tokens, total in beams:
        if tokens[-1] == end_token:
            candidates.append((tokens, total)); continue
        for token, log_probability in enumerate(transition_log_probs[tokens[-1]]):
            candidates.append((tokens + [token], total + float(log_probability)))
    candidates.sort(key=lambda item: (-normalized(*item), item[0]))
    beams = candidates[:beam_width]
    if all(tokens[-1] == end_token for tokens, _ in beams):
        break
beams.sort(key=lambda item: (-normalized(*item), item[0]))
tokens, total = beams[0]
return tokens, float(normalized(tokens, total))
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(290029); cases = []
    for index in range(100):
        vocab = 3 + index % 8; raw = rng.normal(size=(vocab, vocab)); raw -= np.log(np.exp(raw).sum(axis=1, keepdims=True))
        start, end = index % vocab, (index * 3 + 1) % vocab
        width, steps, penalty = 1 + index % min(5, vocab), 1 + index % 7, (0., .6, 1., 1.4)[index % 4]
        cases.append({"name": f"vocab={vocab}, width={width}, alpha={penalty}", "args": (raw, start, end, steps, width, penalty)})
    return cases
'''),
    dict(
        id="30", slug="moe_routing", title="MoE 路由", function="moe_route",
        signature='''\
def moe_route(
    x: np.ndarray,
    gate_weight: np.ndarray,
    expert_w1: np.ndarray,
    expert_w2: np.ndarray,
    top_k: int,
    capacity_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:''',
        background="混合专家层为每个 token 只激活少量专家，从而在计算量近似不变时增加参数容量。路由器需要同时处理 Top-K 选择、门权重和专家容量。",
        requirement="先计算 x@gate_weight 并对专家维做 softmax，再按概率稳定降序选择每个 token 的 top_k 专家，并只在选中项内重新 softmax。每个专家容量为 ceil(capacity_factor*num_tokens*top_k/num_experts)。按 token 顺序、槽位顺序接收路由，超容量项的 assignment 记为 -1 且不贡献输出；其余项通过对应专家的两层 ReLU MLP 并按门权重累加。",
        inputs=[("x", "形状为 (num_tokens, model_dim) 的 float64 token。"), ("gate_weight", "形状为 (model_dim,num_experts) 的路由矩阵。"), ("expert_w1", "形状为 (num_experts,model_dim,hidden_dim)。"), ("expert_w2", "形状为 (num_experts,hidden_dim,model_dim)。"), ("top_k", "每个 token 选择的专家数。"), ("capacity_factor", "大于 0 的容量倍率。")],
        output="返回 `(output, assignments, routing_weights)`；output 与 x 同形，后两者形状为 `(num_tokens,top_k)`，被丢弃的 assignment 为 -1。",
        math="$p=softmax(xW_g)$，选中集合内 $\\tilde p_e=e^{z_e}/\\sum_{j\\in TopK}e^{z_j}$；专家输出为 $ReLU(xW_{1,e})W_{2,e}$。",
        coverage="Top-1/Top-2、不同专家数、容量充足与溢出、概率并列、不同 token/隐藏维度和稳定接收顺序", rtol="1e-8", atol="1e-10",
        answer='''\
logits = x @ gate_weight
order = np.argsort(-logits, axis=1, kind="stable")[:, :top_k]
selected = np.take_along_axis(logits, order, axis=1)
routing_weights = np.exp(selected - selected.max(axis=1, keepdims=True)); routing_weights /= routing_weights.sum(axis=1, keepdims=True)
assignments = order.copy()
num_experts = gate_weight.shape[1]
capacity = int(np.ceil(capacity_factor * x.shape[0] * top_k / num_experts))
counts = np.zeros(num_experts, dtype=int); output = np.zeros_like(x)
for token in range(x.shape[0]):
    for slot in range(top_k):
        expert = int(order[token, slot])
        if counts[expert] >= capacity:
            assignments[token, slot] = -1; continue
        counts[expert] += 1
        hidden = np.maximum(x[token] @ expert_w1[expert], 0.0)
        output[token] += routing_weights[token, slot] * (hidden @ expert_w2[expert])
return output, assignments, routing_weights
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(300030); cases = []
    for index in range(100):
        tokens, dim, experts = 2 + index % 9, 2 + index % 6, 2 + index % 5
        hidden, top_k = 3 + index % 7, 1 + index % min(2, experts)
        x = rng.normal(size=(tokens, dim)); gate = rng.normal(scale=.3, size=(dim, experts)); w1 = rng.normal(scale=.3, size=(experts, dim, hidden)); w2 = rng.normal(scale=.3, size=(experts, hidden, dim)); capacity = (.5, 1., 1.5, 2.)[index % 4]
        cases.append({"name": f"experts={experts}, k={top_k}, capacity={capacity}", "args": (x, gate, w1, w2, top_k, capacity)})
    return cases
'''),
    dict(
        id="31", slug="lora_linear", title="LoRA Linear", function="lora_linear",
        signature='''\
def lora_linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    alpha: float,
    dropout_p: float,
    training: bool,
    seed: int,
) -> np.ndarray:''',
        background="LoRA 冻结原始线性层，仅训练两个低秩矩阵来表示权重增量，显著减少微调参数。缩放和 dropout 的位置会影响与常用实现的一致性。",
        requirement="计算基础分支 x@weight+bias，以及 LoRA 分支 dropout(x)@lora_a@lora_b*(alpha/rank)。dropout 只作用于 LoRA 分支：training=False 或 dropout_p=0 时直接使用 x；训练时用 default_rng(seed) 生成保留掩码并采用 inverted dropout。bias=None 时不加偏置。",
        inputs=[("x", "形状为 (...,in_features) 的 float64 输入。"), ("weight", "形状为 (in_features,out_features) 的冻结权重。"), ("bias", "None 或形状为 (out_features,) 的偏置。"), ("lora_a / lora_b", "形状为 (in_features,rank) 和 (rank,out_features) 的低秩矩阵。"), ("alpha", "LoRA 缩放超参数，实际倍率为 alpha/rank。"), ("dropout_p / training / seed", "丢弃概率、训练标志和可复现随机种子。")],
        output="返回前导维度与 x 相同、末维为 out_features 的 float64 数组。",
        math="$y=xW+b+\\frac{\\alpha}{r}Dropout(x)AB$；训练时 $Dropout(x)=m\\odot x/(1-p)$，其中 $m\\sim Bernoulli(1-p)$。",
        coverage="不同秩和缩放、带/不带偏置、训练/推理、零及非零 dropout、二维/三维输入和固定种子", rtol="1e-9", atol="1e-11",
        answer='''\
base = x @ weight
if bias is not None:
    base = base + bias
adapted = x
if training and dropout_p > 0.0:
    keep = np.random.default_rng(seed).random(x.shape) >= dropout_p
    adapted = x * keep / (1.0 - dropout_p)
rank = lora_a.shape[1]
return base + (adapted @ lora_a @ lora_b) * (alpha / rank)
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(310031); cases = []
    for index in range(100):
        inputs, outputs = 2 + index % 7, 2 + (index * 3) % 8; rank = 1 + index % min(4, inputs, outputs)
        shape = (1 + index % 5, inputs) if index % 3 else (2, 1 + index % 4, inputs)
        x = rng.normal(size=shape); weight = rng.normal(scale=.3, size=(inputs, outputs)); bias = None if index % 4 == 0 else rng.normal(scale=.1, size=outputs); a = rng.normal(scale=.2, size=(inputs, rank)); b = rng.normal(scale=.2, size=(rank, outputs)); alpha = (.5, 1., 4., 8.)[index % 4]; dropout = (0., .1, .25, .5)[index % 4]; training = bool(index % 2)
        cases.append({"name": f"rank={rank}, dropout={dropout}, training={training}", "args": (x, weight, bias, a, b, alpha, dropout, training, 12000 + index)})
    return cases
'''),
]


rich_generate_all(SPECS)
