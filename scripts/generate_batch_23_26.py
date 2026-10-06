from rich_question_codegen import rich_generate_all


SPECS = [
    dict(
        id="23", slug="transformer_ffn", title="Transformer FFN", function="transformer_ffn",
        signature='''\
def transformer_ffn(
    x: np.ndarray,
    w_up: np.ndarray,
    b_up: np.ndarray,
    w_down: np.ndarray,
    b_down: np.ndarray,
    activation: str,
    w_gate: np.ndarray | None = None,
    b_gate: np.ndarray | None = None,
) -> np.ndarray:''',
        background="Transformer 的逐位置前馈网络独立处理每个 token，在模型维度和更宽的隐藏维度之间变换。门控变体把内容分支与激活后的门分支逐元素相乘。",
        requirement="支持 relu、gelu、glu、geglu 和 swiglu。relu/gelu 只使用 up 分支；三个门控模式还使用 gate 分支，其中 glu 的门激活为 sigmoid，geglu 为 GELU，swiglu 为 SiLU。GELU 采用给定的 tanh 近似式。最后统一通过 down 投影。",
        inputs=[("x", "形状为 (..., model_dim) 的 float64 输入。"), ("w_up / b_up", "形状为 (model_dim, hidden_dim) 和 (hidden_dim,) 的内容投影。"), ("w_down / b_down", "形状为 (hidden_dim, model_dim) 和 (model_dim,) 的输出投影。"), ("activation", "relu、gelu、glu、geglu 或 swiglu。"), ("w_gate / b_gate", "门控模式使用的 (model_dim, hidden_dim) 和 (hidden_dim,)；非门控模式为 None。")],
        output="返回与 x 前导维度相同、末维为 model_dim 的 float64 数组。",
        math="$GELU(t)=0.5t[1+\\tanh(\\sqrt{2/\\pi}(t+0.044715t^3))]$，$SiLU(t)=t/(1+e^{-t})$；门控隐藏层为 $up(x)\\odot act(gate(x))$。",
        coverage="五种激活、二维和三维输入、不同模型/隐藏维度、零偏置、正负大值和门控分支", rtol="1e-9", atol="1e-11",
        answer='''\
up = x @ w_up + b_up

def gelu(value):
    return 0.5 * value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value + 0.044715 * value ** 3)))

if activation == "relu":
    hidden = np.maximum(up, 0.0)
elif activation == "gelu":
    hidden = gelu(up)
else:
    gate = x @ w_gate + b_gate
    if activation == "glu":
        gate = 1.0 / (1.0 + np.exp(-gate))
    elif activation == "geglu":
        gate = gelu(gate)
    else:
        gate = gate / (1.0 + np.exp(-gate))
    hidden = up * gate
return hidden @ w_down + b_down
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    modes = ("relu", "gelu", "glu", "geglu", "swiglu")
    for index in range(100):
        model_dim, hidden_dim = 2 + index % 6, 3 + index % 9
        shape = (1 + index % 5, model_dim) if index % 3 else (2, 1 + index % 4, model_dim)
        x = rng.normal(size=shape)
        w_up = rng.normal(scale=0.3, size=(model_dim, hidden_dim)); b_up = rng.normal(scale=0.1, size=hidden_dim)
        w_down = rng.normal(scale=0.3, size=(hidden_dim, model_dim)); b_down = rng.normal(scale=0.1, size=model_dim)
        mode = modes[index % len(modes)]
        if mode in ("relu", "gelu"):
            w_gate = b_gate = None
        else:
            w_gate = rng.normal(scale=0.3, size=(model_dim, hidden_dim)); b_gate = rng.normal(scale=0.1, size=hidden_dim)
        cases.append({"name": f"{mode}, shape={shape}", "args": (x, w_up, b_up, w_down, b_down, mode, w_gate, b_gate)})
    return cases
'''),
    dict(
        id="24", slug="transformer_encoder_block", title="Transformer Encoder Block", function="transformer_encoder_block",
        signature='''\
def transformer_encoder_block(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    pre_norm: bool,
    mask: np.ndarray | None = None,
) -> np.ndarray:''',
        background="编码器块把双向多头自注意力与逐位置前馈网络串联，并通过残差连接和 LayerNorm 稳定深层训练。Pre-Norm 与 Post-Norm 的运算顺序不同，不能混用。",
        requirement="实现单批次编码器块。params 必须使用键 w_q、w_k、w_v、w_o、w1、b1、w2、b2、ln1_scale、ln1_bias、ln2_scale、ln2_bias。注意力使用可选布尔 mask，FFN 使用 tanh 近似 GELU，LayerNorm 在末维计算且 epsilon=1e-5。pre_norm=True 时先归一化再进入子层；否则先残差相加再归一化。",
        inputs=[("x", "形状为 (sequence_length, model_dim) 的 float64 输入。"), ("params", "包含四个注意力矩阵、两层 FFN 参数和两组 LayerNorm 仿射参数的字典。"), ("num_heads", "能整除 model_dim 的注意力头数。"), ("pre_norm", "True 选择 Pre-Norm，False 选择 Post-Norm。"), ("mask", "None 或形状为 (L,L) 的布尔可见性掩码。")],
        output="返回形状与 x 相同的编码器块输出。",
        math="$LN(x)=\\gamma(x-\\mu)/\\sqrt{\\sigma^2+10^{-5}}+\\beta$。Pre-Norm 为 $y=x+Attn(LN_1(x))$、$z=y+FFN(LN_2(y))$；Post-Norm 为对应的残差后归一化。",
        coverage="Pre/Post-Norm、不同头数和序列长度、双向与随机掩码、不同 FFN 宽度及归一化仿射参数", rtol="1e-8", atol="1e-10",
        answer='''\
def layer_norm(value, scale, bias):
    mean = value.mean(axis=-1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
    return (value - mean) / np.sqrt(variance + 1e-5) * scale + bias

def attention(value):
    length, model_dim = value.shape
    head_dim = model_dim // num_heads
    q = (value @ params["w_q"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    k = (value @ params["w_k"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    v = (value @ params["w_v"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    if mask is not None:
        scores = np.where(mask[None], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ v).transpose(1, 0, 2).reshape(length, model_dim) @ params["w_o"]

def ffn(value):
    hidden = value @ params["w1"] + params["b1"]
    hidden = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)))
    return hidden @ params["w2"] + params["b2"]

if pre_norm:
    y = x + attention(layer_norm(x, params["ln1_scale"], params["ln1_bias"]))
    return y + ffn(layer_norm(y, params["ln2_scale"], params["ln2_bias"]))
y = layer_norm(x + attention(x), params["ln1_scale"], params["ln1_bias"])
return layer_norm(y + ffn(y), params["ln2_scale"], params["ln2_bias"])
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(240024)
    cases = []
    for index in range(100):
        heads, head_dim, length = 1 + index % 3, 2 + index % 3, 2 + index % 6
        dim, hidden = heads * head_dim, 2 * heads * head_dim + index % 4
        params = {key: rng.normal(scale=0.25, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}
        params.update(w1=rng.normal(scale=.25, size=(dim, hidden)), b1=rng.normal(scale=.1, size=hidden), w2=rng.normal(scale=.25, size=(hidden, dim)), b2=rng.normal(scale=.1, size=dim), ln1_scale=rng.uniform(.8, 1.2, size=dim), ln1_bias=rng.normal(scale=.1, size=dim), ln2_scale=rng.uniform(.8, 1.2, size=dim), ln2_bias=rng.normal(scale=.1, size=dim))
        x = rng.normal(size=(length, dim)); mask = None if index % 2 else (rng.random((length, length)) > .25); 
        if mask is not None: np.fill_diagonal(mask, True)
        cases.append({"name": f"pre={bool(index % 2)}, heads={heads}", "args": (x, params, heads, bool(index % 2), mask)})
    return cases
'''),
    dict(
        id="25", slug="transformer_decoder_block", title="Transformer Decoder Block", function="transformer_decoder_block",
        signature='''\
def transformer_decoder_block(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    position_ids: np.ndarray,
) -> np.ndarray:''',
        background="现代 Decoder-only 模型通常组合因果自注意力、RoPE、RMSNorm 和 SwiGLU。本题把这些组件连接成一个可独立验证的 Pre-Norm 解码器块。",
        requirement="params 包含 w_q、w_k、w_v、w_o、w_gate、w_up、w_down、norm1_scale、norm2_scale。先 RMSNorm，再对投影后的全部 Q/K 头维应用 RoPE，使用下三角因果掩码；完成第一条残差后再次 RMSNorm，并通过 SwiGLU FFN 和第二条残差。RMSNorm epsilon=1e-6。",
        inputs=[("x", "形状为 (sequence_length, model_dim) 的 float64 隐状态。"), ("params", "注意力、SwiGLU 和两组 RMSNorm 参数字典；投影不含偏置。"), ("num_heads", "能整除模型维度的头数，头维为偶数。"), ("position_ids", "形状为 (L,) 的非负绝对位置。")],
        output="返回形状为 `(L, model_dim)` 的解码器块输出。",
        math="$RMSNorm(x)=x/\\sqrt{mean(x^2)+10^{-6}}\\odot g$；RoPE 旋转后使用因果注意力；$SwiGLU(x)=SiLU(xW_g)\\odot(xW_u)$。",
        coverage="不同长度和起始位置、多个头数与偶数头维、因果边界、不同 FFN 宽度及 RMSNorm 缩放", rtol="1e-8", atol="1e-10",
        answer='''\
def rms_norm(value, scale):
    return value / np.sqrt(np.mean(value ** 2, axis=-1, keepdims=True) + 1e-6) * scale

length, model_dim = x.shape
head_dim = model_dim // num_heads
normalized = rms_norm(x, params["norm1_scale"])
q = (normalized @ params["w_q"]).reshape(length, num_heads, head_dim)
k = (normalized @ params["w_k"]).reshape(length, num_heads, head_dim)
v = (normalized @ params["w_v"]).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
frequencies = 10000.0 ** (-np.arange(0, head_dim, 2) / head_dim)
angles = position_ids[:, None] * frequencies[None, :]
cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
def rotate(value):
    result = np.empty_like(value); even, odd = value[..., 0::2], value[..., 1::2]
    result[..., 0::2] = even * cosine - odd * sine; result[..., 1::2] = even * sine + odd * cosine
    return result.transpose(1, 0, 2)
q, k = rotate(q), rotate(k)
scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
scores = np.where(np.tril(np.ones((length, length), dtype=bool))[None], scores, -np.inf)
weights = np.exp(scores - scores.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
attention = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim) @ params["w_o"]
hidden = x + attention
normalized = rms_norm(hidden, params["norm2_scale"])
gate = normalized @ params["w_gate"]; up = normalized @ params["w_up"]
ffn = (gate / (1.0 + np.exp(-gate)) * up) @ params["w_down"]
return hidden + ffn
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(250025); cases = []
    for index in range(100):
        heads, head_dim, length = 1 + index % 3, (2, 4, 6)[index % 3], 1 + index % 7
        dim, hidden = heads * head_dim, 2 * heads * head_dim + index % 5
        params = {key: rng.normal(scale=.25, size=(dim, dim)) for key in ("w_q", "w_k", "w_v", "w_o")}
        params.update(w_gate=rng.normal(scale=.25, size=(dim, hidden)), w_up=rng.normal(scale=.25, size=(dim, hidden)), w_down=rng.normal(scale=.25, size=(hidden, dim)), norm1_scale=rng.uniform(.8, 1.2, size=dim), norm2_scale=rng.uniform(.8, 1.2, size=dim))
        x = rng.normal(size=(length, dim)); positions = np.arange(length) + index % 17
        cases.append({"name": f"heads={heads}, start={positions[0]}", "args": (x, params, heads, positions)})
    return cases
'''),
    dict(
        id="26", slug="encoder_decoder_block", title="Encoder-Decoder Block", function="encoder_decoder_block",
        signature='''\
def encoder_decoder_block(
    decoder_x: np.ndarray,
    encoder_x: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    self_mask: np.ndarray | None = None,
    cross_mask: np.ndarray | None = None,
) -> np.ndarray:''',
        background="序列到序列 Transformer 的解码器既要读取已生成的目标前缀，也要读取编码器表示。因此一个块包含因果自注意力、交叉注意力和前馈网络三个子层。",
        requirement="实现三个 Pre-LayerNorm 子层及各自残差。params 用 self_q/self_k/self_v/self_o 表示自注意力矩阵，cross_q/cross_k/cross_v/cross_o 表示交叉注意力矩阵，ff1/ff1_bias/ff2/ff2_bias 表示 GELU FFN，并提供 ln1/ln2/ln3 的 scale 和 bias。self_mask 与 cross_mask 中 True 表示可见，LayerNorm epsilon=1e-5。",
        inputs=[("decoder_x", "形状为 (target_length, model_dim) 的解码器状态。"), ("encoder_x", "形状为 (source_length, model_dim) 的编码器输出。"), ("params", "三子层投影与 LayerNorm 参数字典。"), ("num_heads", "能整除 model_dim 的头数。"), ("self_mask", "None 或 (T,T) 布尔掩码。"), ("cross_mask", "None 或 (T,S) 布尔掩码。")],
        output="返回形状为 `(target_length, model_dim)` 的更新后解码器状态。",
        math="$y=x+SelfAttn(LN_1(x))$，$z=y+CrossAttn(LN_2(y),encoder)$，$o=z+FFN(LN_3(z))$；交叉注意力的 K/V 只来自 encoder_x。",
        coverage="不同源/目标长度、多个头数、因果与随机交叉掩码、单 token 解码、不同 FFN 宽度和归一化参数", rtol="1e-8", atol="1e-10",
        answer='''\
def norm(value, prefix):
    mean = value.mean(axis=-1, keepdims=True); var = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
    return (value - mean) / np.sqrt(var + 1e-5) * params[prefix + "_scale"] + params[prefix + "_bias"]

def attend(query, context, prefix, mask):
    q_len, dim = query.shape; k_len = context.shape[0]; head_dim = dim // num_heads
    q = (query @ params[prefix + "_q"]).reshape(q_len, num_heads, head_dim).transpose(1, 0, 2)
    k = (context @ params[prefix + "_k"]).reshape(k_len, num_heads, head_dim).transpose(1, 0, 2)
    v = (context @ params[prefix + "_v"]).reshape(k_len, num_heads, head_dim).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    if mask is not None: scores = np.where(mask[None], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True)); weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ v).transpose(1, 0, 2).reshape(q_len, dim) @ params[prefix + "_o"]

y_input = norm(decoder_x, "ln1"); y = decoder_x + attend(y_input, y_input, "self", self_mask)
z = y + attend(norm(y, "ln2"), encoder_x, "cross", cross_mask)
hidden = norm(z, "ln3") @ params["ff1"] + params["ff1_bias"]
hidden = .5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + .044715 * hidden ** 3)))
return z + hidden @ params["ff2"] + params["ff2_bias"]
''',
        cases='''\
def build_cases():
    rng = np.random.default_rng(260026); cases = []
    for index in range(100):
        heads, head_dim = 1 + index % 3, 2 + index % 3; dim = heads * head_dim
        target_len, source_len, hidden = 1 + index % 5, 2 + index % 7, 2 * dim + index % 3
        params = {}
        for prefix in ("self", "cross"):
            for suffix in ("q", "k", "v", "o"): params[prefix + "_" + suffix] = rng.normal(scale=.2, size=(dim, dim))
        params.update(ff1=rng.normal(scale=.2, size=(dim, hidden)), ff1_bias=rng.normal(scale=.1, size=hidden), ff2=rng.normal(scale=.2, size=(hidden, dim)), ff2_bias=rng.normal(scale=.1, size=dim))
        for prefix in ("ln1", "ln2", "ln3"):
            params[prefix + "_scale"] = rng.uniform(.8, 1.2, size=dim); params[prefix + "_bias"] = rng.normal(scale=.1, size=dim)
        decoder_x, encoder_x = rng.normal(size=(target_len, dim)), rng.normal(size=(source_len, dim))
        self_mask = np.tril(np.ones((target_len, target_len), dtype=bool)); cross_mask = rng.random((target_len, source_len)) > .2; cross_mask[:, 0] = True
        cases.append({"name": f"target={target_len}, source={source_len}", "args": (decoder_x, encoder_x, params, heads, self_mask, cross_mask)})
    return cases
'''),
]


rich_generate_all(SPECS)
