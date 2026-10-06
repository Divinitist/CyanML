from question_codegen import generate_all


SPECS = [
    {
        "id": "19", "slug": "absolute_position_encoding", "title": "绝对位置编码",
        "function": "add_sinusoidal_position_encoding",
        "signature": '''\
def add_sinusoidal_position_encoding(x, start_position=0, base=10000.0):''',
        "problem": '''\
# 绝对位置编码

实现 Transformer 正弦绝对位置编码并加到 `x` 上。`x` 形状为 `(序列长度, 模型维度)`；偶数维使用 sin，奇数维使用 cos，角频率为 `base^(-2i/d_model)`。`start_position` 用于分块序列的起始位置。
''',
        "answer": '''\
length, model_dim = x.shape
positions = (start_position + np.arange(length, dtype=np.float64))[:, None]
even_indices = np.arange(0, model_dim, 2)
angles = positions / (base ** (even_indices / model_dim))[None, :]
encoding = np.zeros_like(x, dtype=np.float64)
encoding[:, 0::2] = np.sin(angles)
odd_count = encoding[:, 1::2].shape[1]
encoding[:, 1::2] = np.cos(angles[:, :odd_count])
return x + encoding
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length = 1 + index % 12
        model_dim = 1 + index % 13
        x = rng.normal(size=(length, model_dim)).astype(np.float64)
        start = (index * 7) % 101
        base = (100.0, 1000.0, 10000.0)[index % 3]
        cases.append({"name": f"shape={x.shape}, start={start}", "args": (x, start, base)})
    return cases
''', "rtol": "1e-10", "atol": "1e-12",
    },
    {
        "id": "20", "slug": "rope", "title": "RoPE",
        "function": "apply_rope",
        "signature": '''\
def apply_rope(q, k, position_ids, base=10000.0, rotary_dim=None):''',
        "problem": '''\
# RoPE

为 Q 和 K 应用旋转位置编码。`q`、`k` 形状为 `(序列长度, 头数, 头维度)`；对前 `rotary_dim` 个维度按相邻偶奇维成对旋转，其余维度保持不变。`rotary_dim=None` 表示旋转全部维度，且旋转维数保证为偶数。
''',
        "answer": '''\
rotary_dim = q.shape[-1] if rotary_dim is None else rotary_dim
frequencies = base ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
angles = position_ids[:, None] * frequencies[None, :]
cosine = np.cos(angles)[:, None, :]
sine = np.sin(angles)[:, None, :]

def rotate(value):
    result = value.copy()
    even = value[..., :rotary_dim:2]
    odd = value[..., 1:rotary_dim:2]
    result[..., :rotary_dim:2] = even * cosine - odd * sine
    result[..., 1:rotary_dim:2] = even * sine + odd * cosine
    return result

return rotate(q), rotate(k)
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length, heads = 1 + index % 9, 1 + index % 4
        head_dim = (2, 4, 6, 8)[index % 4]
        rotary_dim = (2 * (1 + index % (head_dim // 2)))
        q = rng.normal(size=(length, heads, head_dim))
        k = rng.normal(size=(length, heads, head_dim))
        positions = np.arange(length, dtype=np.float64) + index % 11
        cases.append({"name": f"dim={head_dim}, rotary={rotary_dim}", "args": (q, k, positions, 10000.0, rotary_dim)})
    return cases
''',
    },
    {
        "id": "21", "slug": "long_context_rope", "title": "长上下文 RoPE",
        "function": "apply_scaled_rope",
        "signature": '''\
def apply_scaled_rope(q, k, position_ids, scale, method, base=10000.0):''',
        "problem": '''\
# 长上下文 RoPE

实现两种 RoPE 上下文扩展：`method="linear"` 时将位置除以 `scale`；`method="ntk"` 时保持位置不变并将基数调整为 `base * scale^(d/(d-2))`。Q/K 形状为 `(序列长度, 头数, 偶数头维度)`，返回旋转后的 Q/K。
''',
        "answer": '''\
head_dim = q.shape[-1]
if method == "linear":
    positions = position_ids / scale
    adjusted_base = base
else:
    positions = position_ids
    adjusted_base = base * scale ** (head_dim / (head_dim - 2))
frequencies = adjusted_base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
angles = positions[:, None] * frequencies[None, :]
cosine = np.cos(angles)[:, None, :]
sine = np.sin(angles)[:, None, :]

def rotate(value):
    result = np.empty_like(value)
    even, odd = value[..., 0::2], value[..., 1::2]
    result[..., 0::2] = even * cosine - odd * sine
    result[..., 1::2] = even * sine + odd * cosine
    return result

return rotate(q), rotate(k)
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length, heads = 1 + index % 8, 1 + index % 3
        head_dim = (4, 6, 8, 10)[index % 4]
        q = rng.normal(size=(length, heads, head_dim))
        k = rng.normal(size=(length, heads, head_dim))
        positions = np.arange(length, dtype=np.float64) + index * 13
        scale = (1.25, 2.0, 4.0, 8.0)[index % 4]
        method = "linear" if index % 2 == 0 else "ntk"
        cases.append({"name": f"{method}, scale={scale}", "args": (q, k, positions, scale, method, 10000.0)})
    return cases
''',
    },
    {
        "id": "22", "slug": "relative_position_encoding", "title": "相对位置编码",
        "function": "relative_position_bias",
        "signature": '''\
def relative_position_bias(query_length, key_length, num_heads, mode, max_distance, table=None):''',
        "problem": '''\
# 相对位置编码

生成形状为 `(头数, Q长度, K长度)` 的相对位置偏置。`mode="alibi"` 使用每个头斜率 `2^(-8(h+1)/H)` 乘以负绝对距离；`mode="table"` 将有符号距离 `key-query` 截断到 `[-max_distance,max_distance]` 后查询形状为 `(H, 2*max_distance+1)` 的表。
''',
        "answer": '''\
query_positions = np.arange(query_length)[:, None]
key_positions = np.arange(key_length)[None, :]
relative = key_positions - query_positions
if mode == "alibi":
    slopes = 2.0 ** (-8.0 * (np.arange(num_heads) + 1) / num_heads)
    return -slopes[:, None, None] * np.abs(relative)[None, :, :]
indices = np.clip(relative, -max_distance, max_distance) + max_distance
return table[:, indices]
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        query_length = 1 + index % 9
        key_length = 1 + (index * 3) % 11
        heads = 1 + index % 6
        maximum = 1 + index % 5
        mode = "alibi" if index % 2 == 0 else "table"
        table = None if mode == "alibi" else rng.normal(size=(heads, 2 * maximum + 1))
        cases.append({"name": f"{mode} {query_length}x{key_length}", "args": (query_length, key_length, heads, mode, maximum, table)})
    return cases
''',
    },
]


generate_all(SPECS)
