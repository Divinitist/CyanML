from rich_question_codegen import rich_generate_all


SPECS = [
    dict(
        id="32", slug="numerical_gradient_checker", title="数值梯度检查器", function="check_gradient",
        signature='''\
def check_gradient(
    function: callable,
    x: np.ndarray,
    analytic_gradient: np.ndarray,
    epsilon: float,
    tolerance: float,
) -> tuple[np.ndarray, float, bool]:''',
        background="手写反向传播时，中心差分可以用纯前向计算近似每个参数的偏导，从而检查解析梯度。它速度较慢，但非常适合定位小规模实现中的链式法则错误。",
        requirement="对 x 的每个元素分别加减 epsilon，调用返回标量 float 的 function，并用中心差分计算数值梯度。每次扰动后必须恢复原值，且不得修改调用者传入的 x。计算解析梯度与数值梯度的逐元素相对误差，分母为 max(1,|a|,|n|)，返回最大误差及其是否不超过 tolerance。",
        inputs=[("function", "接收一个与 x 同形数组并返回标量的纯函数。"), ("x", "任意形状的小型 float64 数组。"), ("analytic_gradient", "与 x 同形的解析梯度。"), ("epsilon", "中心差分的正扰动步长。"), ("tolerance", "最大相对误差阈值。")],
        output="返回 `(numerical_gradient, max_relative_error, passed)`；梯度与 x 同形，误差为 Python float，passed 为 Python bool。",
        math="$g_i^{num}=[f(x+\\epsilon e_i)-f(x-\\epsilon e_i)]/(2\\epsilon)$；相对误差为 $|g_i^{num}-g_i|/max(1,|g_i^{num}|,|g_i|)$。",
        coverage="线性、平方、三次、sin、exp 和矩阵二次型，不同数组形状、步长、正确梯度及刻意偏移的错误梯度", rtol="1e-9", atol="1e-11",
        answer='''\
working = x.copy()
numerical = np.empty_like(working, dtype=np.float64)
for index in np.ndindex(working.shape):
    original = working[index]
    working[index] = original + epsilon; plus = float(function(working))
    working[index] = original - epsilon; minus = float(function(working))
    working[index] = original
    numerical[index] = (plus - minus) / (2.0 * epsilon)
denominator = np.maximum(1.0, np.maximum(np.abs(numerical), np.abs(analytic_gradient)))
max_error = float(np.max(np.abs(numerical - analytic_gradient) / denominator))
return numerical, max_error, bool(max_error <= tolerance)
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(320032); cases = []
    for index in range(100):
        shape = (1 + index % 7,) if index % 3 else (2 + index % 3, 1 + index % 4)
        x = rng.normal(scale=.5, size=shape); mode = index % 5
        if mode == 0: function = lambda value: np.sum(value ** 2); gradient = 2 * x
        elif mode == 1: function = lambda value: np.sum(value ** 3); gradient = 3 * x ** 2
        elif mode == 2: function = lambda value: np.sum(np.sin(value)); gradient = np.cos(x)
        elif mode == 3: function = lambda value: np.sum(np.exp(value)); gradient = np.exp(x)
        else:
            coefficient = rng.normal(size=shape); function = lambda value, c=coefficient: np.sum(c * value); gradient = coefficient.copy()
        if index % 10 == 0: gradient = gradient + 1e-3
        epsilon, tolerance = (1e-5, 1e-6) if index % 2 else (1e-6, 1e-5)
        cases.append({"name": f"mode={mode}, shape={shape}", "args": (function, x, gradient, epsilon, tolerance), "input": {"mode": mode, "shape": shape, "epsilon": epsilon, "tolerance": tolerance}})
    return cases
'''),
    dict(
        id="33", slug="linear_backward", title="Linear 完整 backward", function="linear_forward_backward",
        signature='''\
def linear_forward_backward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    grad_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:''',
        background="全连接层的反向传播是理解自动微分的基础。一个上游梯度需要同时沿计算图传播到输入、权重和偏置，并对批次维正确求和。",
        requirement="前向计算 output=x@weight+bias，其中 x 为二维批次矩阵。给定与 output 同形的 grad_output，使用矩阵微分公式一次性计算 grad_x、grad_weight 和 grad_bias。函数只返回计算结果，不在原数组上累积梯度。",
        inputs=[("x", "形状为 (batch_size,in_features) 的 float64 输入。"), ("weight", "形状为 (in_features,out_features) 的权重。"), ("bias", "形状为 (out_features,) 的偏置。"), ("grad_output", "形状为 (batch_size,out_features) 的上游梯度。")],
        output="返回 `(output, grad_x, grad_weight, grad_bias)`，形状依次为 `(N,O)`、`(N,I)`、`(I,O)`、`(O,)`。",
        math="$Y=XW+b$，$\\partial L/\\partial X=G W^T$，$\\partial L/\\partial W=X^T G$，$\\partial L/\\partial b=\\sum_nG_n$。",
        coverage="单/多样本、不同输入输出维度、零值、负值、不同数值尺度，并通过独立数值差分覆盖所有梯度方向", rtol="1e-10", atol="1e-12",
        answer='''\
output = x @ weight + bias
grad_x = grad_output @ weight.T
grad_weight = x.T @ grad_output
grad_bias = np.sum(grad_output, axis=0)
return output, grad_x, grad_weight, grad_bias
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(330033); cases = []
    for index in range(100):
        batch, inputs, outputs = 1 + index % 7, 1 + (index * 2) % 8, 1 + (index * 3) % 9
        scale = (0.01, .1, 1., 3.)[index % 4]
        x = rng.normal(scale=scale, size=(batch, inputs)); weight = rng.normal(scale=scale, size=(inputs, outputs)); bias = rng.normal(scale=scale, size=outputs); grad = rng.normal(size=(batch, outputs))
        cases.append({"name": f"{batch}x{inputs}->{outputs}, scale={scale}", "args": (x, weight, bias, grad)})
    return cases
'''),
    dict(
        id="34", slug="embedding_lm_head", title="Embedding + LM Head", function="embedding_lm_head",
        signature='''\
def embedding_lm_head(
    token_ids: np.ndarray,
    embedding_weight: np.ndarray,
    hidden_projection: np.ndarray,
    lm_head_weight: np.ndarray,
    lm_head_bias: np.ndarray | None,
    tie_weights: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:''',
        background="语言模型先把离散 token 下标查表为向量，经过隐藏变换后再由 LM Head 投影回词表。输入嵌入与输出权重共享可以减少参数，并让两个空间直接对齐。",
        requirement="用 token_ids 对 embedding_weight 做 NumPy 高级索引得到 embeddings；hidden_states=tanh(embeddings@hidden_projection)。tie_weights=True 时忽略 lm_head_weight，使用 embedding_weight 作为输出矩阵；否则使用 lm_head_weight。LM Head 按 hidden_states@output_weight.T 计算，并在 bias 非 None 时相加。",
        inputs=[("token_ids", "任意前导形状的整数 token 数组，元素位于词表范围内。"), ("embedding_weight", "形状为 (vocab_size,model_dim) 的查表矩阵。"), ("hidden_projection", "形状为 (model_dim,model_dim) 的隐藏变换。"), ("lm_head_weight", "未共享时使用的 (vocab_size,model_dim) 权重。"), ("lm_head_bias", "None 或形状为 (vocab_size,) 的偏置。"), ("tie_weights", "是否共享输入嵌入权重。")],
        output="返回 `(embeddings, hidden_states, logits)`，前两者末维为 model_dim，logits 末维为 vocab_size。",
        math="$E=W_{emb}[ids]$，$H=\\tanh(EW_h)$，$logits=HW_{out}^T+b$；共享时 $W_{out}=W_{emb}$。",
        coverage="一维/二维 token、重复 token、共享/独立权重、有/无偏置、不同词表和模型维度", rtol="1e-10", atol="1e-12",
        answer='''\
embeddings = embedding_weight[token_ids]
hidden_states = np.tanh(embeddings @ hidden_projection)
output_weight = embedding_weight if tie_weights else lm_head_weight
logits = hidden_states @ output_weight.T
if lm_head_bias is not None:
    logits = logits + lm_head_bias
return embeddings, hidden_states, logits
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(340034); cases = []
    for index in range(100):
        vocab, dim = 3 + index % 17, 2 + index % 8
        shape = (1 + index % 8,) if index % 3 else (2, 1 + index % 5)
        ids = rng.integers(0, vocab, size=shape); embedding = rng.normal(scale=.3, size=(vocab, dim)); projection = rng.normal(scale=.3, size=(dim, dim)); head = rng.normal(scale=.3, size=(vocab, dim)); bias = None if index % 4 == 0 else rng.normal(scale=.1, size=vocab); tied = bool(index % 2)
        cases.append({"name": f"shape={shape}, tied={tied}, bias={bias is not None}", "args": (ids, embedding, projection, head, bias, tied)})
    return cases
'''),
    dict(
        id="35", slug="dropout", title="Dropout", function="dropout",
        signature='''\
def dropout(
    x: np.ndarray,
    probability: float,
    training: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:''',
        background="Dropout 在训练时随机屏蔽激活以减少共适应，推理时保持所有激活。Inverted Dropout 在训练阶段完成缩放，因此推理阶段无需额外乘系数。",
        requirement="training=True 且 probability>0 时，用 default_rng(seed).random(x.shape) >= probability 生成布尔保留掩码，输出 x*mask/(1-probability)。推理模式或 probability=0 时返回 x 的副本和全 True 掩码。不得修改 x。",
        inputs=[("x", "任意形状的 float64 输入数组。"), ("probability", "满足 0<=p<1 的丢弃概率。"), ("training", "True 表示训练模式，False 表示推理模式。"), ("seed", "生成掩码的整数随机种子。")],
        output="返回 `(output, keep_mask)`；output 与 x 同形，keep_mask 是同形布尔数组。",
        math="训练时 $y_i=x_i m_i/(1-p)$，$m_i\\sim Bernoulli(1-p)$，因此对随机掩码取期望有 $E[y_i]=x_i$。",
        coverage="标量风格小数组及多维数组、p=0、高丢弃率、训练/推理、全零输入、不同随机种子和复现性", rtol="1e-12", atol="1e-12",
        answer='''\
if not training or probability == 0.0:
    return x.copy(), np.ones(x.shape, dtype=bool)
mask = np.random.default_rng(seed).random(x.shape) >= probability
return x * mask / (1.0 - probability), mask
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(350035); cases = []
    for index in range(100):
        shape = ((1 + index % 11,), (2 + index % 4, 1 + index % 7), (2, 3, 1 + index % 5))[index % 3]
        x = rng.normal(size=shape); probability = (0., .1, .25, .5, .8)[index % 5]; training = bool(index % 2)
        if index % 13 == 0: x.fill(0.)
        cases.append({"name": f"shape={shape}, p={probability}, training={training}", "args": (x, probability, training, 15000 + index)})
    return cases
'''),
    dict(
        id="36", slug="decoder_only_transformer", title="Decoder-only Transformer", function="decoder_only_transformer",
        signature='''\
def decoder_only_transformer(
    token_ids: np.ndarray,
    embedding_weight: np.ndarray,
    blocks: list[dict[str, np.ndarray]],
    final_norm_scale: np.ndarray,
    num_heads: int,
) -> np.ndarray:''',
        background="Decoder-only Transformer 把 token embedding、若干因果解码器块、最终归一化和词表投影连接成完整前向过程。本题综合前面的 RMSNorm、RoPE、注意力和 SwiGLU。",
        requirement="输入是单条序列。每个 blocks 元素包含 w_q/w_k/w_v/w_o、w_gate/w_up/w_down、norm1_scale/norm2_scale；所有线性层无偏置。每层使用 Pre-RMSNorm、对 Q/K 全头维施加 RoPE、下三角因果掩码、注意力残差、SwiGLU 残差。最后做 RMSNorm，并与 embedding_weight 共享 LM Head 权重。epsilon=1e-6，位置从 0 开始。",
        inputs=[("token_ids", "形状为 (sequence_length,) 的整数 token 序列。"), ("embedding_weight", "形状为 (vocab_size,model_dim) 的输入/输出共享权重。"), ("blocks", "一个或多个解码器块参数字典组成的列表。"), ("final_norm_scale", "形状为 (model_dim,) 的最终 RMSNorm 缩放。"), ("num_heads", "能整除 model_dim 且头维为偶数的头数。")],
        output="返回形状为 `(sequence_length,vocab_size)` 的 next-token logits。",
        math="每层为 $h'=h+Attn(RMSNorm(h))$、$h''=h'+SwiGLU(RMSNorm(h'))$；最终 $logits=RMSNorm(h'')W_{emb}^T$。",
        coverage="不同词表/序列长度、一至两层、不同头数和偶数头维、不同 FFN 宽度、重复 token 与单 token 序列", rtol="1e-8", atol="1e-10",
        answer='''\
def rms(value, scale):
    return value / np.sqrt(np.mean(value ** 2, axis=-1, keepdims=True) + 1e-6) * scale

hidden = embedding_weight[token_ids].copy()
length, dim = hidden.shape; head_dim = dim // num_heads
positions = np.arange(length); frequencies = 10000.0 ** (-np.arange(0, head_dim, 2) / head_dim)
angles = positions[:, None] * frequencies[None, :]; cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
for params in blocks:
    normalized = rms(hidden, params["norm1_scale"])
    q = (normalized @ params["w_q"]).reshape(length, num_heads, head_dim); k = (normalized @ params["w_k"]).reshape(length, num_heads, head_dim); v = (normalized @ params["w_v"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    def rotate(value):
        result = np.empty_like(value); even, odd = value[..., 0::2], value[..., 1::2]
        result[..., 0::2] = even * cosine - odd * sine; result[..., 1::2] = even * sine + odd * cosine
        return result.transpose(1, 0, 2)
    q, k = rotate(q), rotate(k); scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    scores = np.where(np.tril(np.ones((length, length), dtype=bool))[None], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
    hidden = hidden + (weights @ v).transpose(1, 0, 2).reshape(length, dim) @ params["w_o"]
    normalized = rms(hidden, params["norm2_scale"]); gate = normalized @ params["w_gate"]; up = normalized @ params["w_up"]
    hidden = hidden + (gate / (1 + np.exp(-gate)) * up) @ params["w_down"]
return rms(hidden, final_norm_scale) @ embedding_weight.T
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(360036); cases = []
    for index in range(100):
        heads, head_dim = 1 + index % 3, (2, 4, 6)[index % 3]; dim = heads * head_dim
        vocab, length, layers = max(5, dim + 2 + index % 7), 1 + index % 6, 1 + index % 2
        embedding = rng.normal(scale=.25, size=(vocab, dim)); blocks = []
        for _ in range(layers):
            hidden = 2 * dim + index % 4; params = {key: rng.normal(scale=.2, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}; params.update(w_gate=rng.normal(scale=.2, size=(dim, hidden)), w_up=rng.normal(scale=.2, size=(dim, hidden)), w_down=rng.normal(scale=.2, size=(hidden, dim)), norm1_scale=rng.uniform(.8, 1.2, size=dim), norm2_scale=rng.uniform(.8, 1.2, size=dim)); blocks.append(params)
        ids = rng.integers(0, vocab, size=length); final_scale = rng.uniform(.8, 1.2, size=dim)
        cases.append({"name": f"length={length}, layers={layers}, heads={heads}", "args": (ids, embedding, blocks, final_scale, heads)})
    return cases
'''),
    dict(
        id="37", slug="autoregressive_generation", title="自回归生成", function="autoregressive_generate",
        signature='''\
def autoregressive_generate(
    prompt_tokens: np.ndarray,
    embedding_weight: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    max_new_tokens: int,
    end_token: int,
    use_cache: bool,
) -> tuple[list[int], np.ndarray]:''',
        background="自回归生成反复取最后位置 logits 选择下一个 token。缓存模式只计算新 token 的 Q/K/V，非缓存模式每步重算整个前缀；在没有随机采样时，两条路径应产生相同结果。",
        requirement="实现一个单层因果自注意力语言模型的贪心生成。params 包含 w_q/w_k/w_v/w_o，注意力输出与 embedding 残差相加，再与 embedding_weight 共享 LM Head。每步选择 argmax（并列取较小下标），生成 end_token 时立即停止。use_cache=True 时预填充 prompt 并追加 KV；False 时每步重算完整序列。返回每个决策步骤使用的 logits。",
        inputs=[("prompt_tokens", "形状为 (prompt_length,) 的非空整数前缀。"), ("embedding_weight", "形状为 (vocab_size,model_dim) 的共享嵌入/LM Head。"), ("params", "含四个 (model_dim,model_dim) 注意力矩阵的字典。"), ("num_heads", "能整除 model_dim 的查询/KV 头数。"), ("max_new_tokens", "最多生成的 token 数。"), ("end_token", "生成后提前停止的合法 token 下标。"), ("use_cache", "是否启用 KV Cache 增量路径。")],
        output="返回 `(tokens, step_logits)`；tokens 是包含原 prompt 和新 token 的列表，step_logits 形状为 `(实际生成步数,vocab_size)`。",
        math="每步 $t_{next}=argmax(logits_{last})$。缓存路径保存 $K_{1:t},V_{1:t}$，新查询只与已缓存键值计算；它应与完整因果前向的最后位置相等。",
        coverage="缓存/非缓存、单/多 token prompt、不同生成上限、提前 EOS、多个头数、不同词表以及相同参数下两路径一致性", rtol="1e-8", atol="1e-10",
        answer='''\
tokens = [int(token) for token in prompt_tokens]
head_dim = embedding_weight.shape[1] // num_heads

def attend(chunk_tokens, cache_k=None, cache_v=None):
    x = embedding_weight[np.asarray(chunk_tokens)]; chunk, dim = x.shape
    q = (x @ params["w_q"]).reshape(chunk, num_heads, head_dim).transpose(1, 0, 2)
    new_k = (x @ params["w_k"]).reshape(chunk, num_heads, head_dim).transpose(1, 0, 2)
    new_v = (x @ params["w_v"]).reshape(chunk, num_heads, head_dim).transpose(1, 0, 2)
    past = 0 if cache_k is None else cache_k.shape[1]
    all_k = new_k if cache_k is None else np.concatenate((cache_k, new_k), axis=1)
    all_v = new_v if cache_v is None else np.concatenate((cache_v, new_v), axis=1)
    scores = q @ all_k.transpose(0, 2, 1) / np.sqrt(head_dim)
    mask = np.arange(all_k.shape[1])[None, :] <= past + np.arange(chunk)[:, None]
    scores = np.where(mask[None], scores, -np.inf); weights = np.exp(scores - scores.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
    hidden = x + (weights @ all_v).transpose(1, 0, 2).reshape(chunk, dim) @ params["w_o"]
    return hidden @ embedding_weight.T, all_k, all_v

step_logits = []
cache_k = cache_v = None
if use_cache:
    logits, cache_k, cache_v = attend(tokens)
for _ in range(max_new_tokens):
    if not use_cache:
        logits, _, _ = attend(tokens)
    current = logits[-1].copy(); step_logits.append(current); next_token = int(np.argmax(current)); tokens.append(next_token)
    if next_token == end_token: break
    if use_cache:
        logits, cache_k, cache_v = attend([next_token], cache_k, cache_v)
return tokens, np.stack(step_logits) if step_logits else np.empty((0, embedding_weight.shape[0]))
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(370037); cases = []
    shared = []
    for pair in range(50):
        heads, head_dim = 1 + pair % 3, 2 + pair % 3; dim = heads * head_dim; vocab = max(5, dim + 3 + pair % 5); prompt_length = 1 + pair % 5
        embedding = rng.normal(scale=.3, size=(vocab, dim)); params = {key: rng.normal(scale=.2, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}; prompt = rng.integers(0, vocab, size=prompt_length); steps, end = 1 + pair % 5, pair % vocab
        shared.append((prompt, embedding, params, heads, steps, end))
    for index in range(100):
        common = shared[index // 2]; cached = bool(index % 2)
        cases.append({"name": f"pair={index // 2}, cache={cached}", "args": (*common, cached)})
    return cases
'''),
]


rich_generate_all(SPECS)
