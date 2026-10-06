from question_codegen import generate_all


COMMON_CASE_PREFIX = '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length = 2 + index % 7
        heads = 1 + index % 4
        head_dim = 2 + index % 3
        model_dim = heads * head_dim
        x = rng.normal(size=(length, model_dim)).astype(np.float64)
'''


SPECS = [
    {
        "id": "14", "slug": "multi_head_attention", "title": "Multi-Head Attention",
        "function": "multi_head_attention",
        "signature": '''\
def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, mask=None):''',
        "problem": '''\
# Multi-Head Attention

实现 `multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, mask=None)`。输入形状为 `(序列长度, 模型维度)`，完成 Q/K/V 线性投影、分头、缩放点积注意力、拼接和输出投影。布尔 `mask` 形状为 `(Q长度, K长度)`，`True` 表示允许关注。使用 float64；误差 `rtol=1e-9, atol=1e-11`。
''',
        "answer": '''\
length, model_dim = x.shape
head_dim = model_dim // num_heads
q = (x @ w_q).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
k = (x @ w_k).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
v = (x @ w_v).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
if mask is not None:
    scores = np.where(mask[None, :, :], scores, -np.inf)
maximum = np.max(scores, axis=-1, keepdims=True)
weights = np.exp(scores - maximum)
weights /= np.sum(weights, axis=-1, keepdims=True)
context = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim)
return context @ w_o
''',
        "cases": COMMON_CASE_PREFIX + '''\
        weights = [rng.normal(scale=0.3, size=(model_dim, model_dim)) for _ in range(4)]
        mask = np.tril(np.ones((length, length), dtype=bool)) if index % 2 else None
        cases.append({"name": f"heads={heads}, masked={mask is not None}", "args": (x, *weights, heads, mask)})
    return cases
''', "rtol": "1e-9", "atol": "1e-11",
    },
    {
        "id": "15", "slug": "multi_query_attention", "title": "Multi-Query Attention",
        "function": "multi_query_attention",
        "signature": '''\
def multi_query_attention(x, w_q, w_k, w_v, w_o, num_query_heads, mask=None):''',
        "problem": '''\
# Multi-Query Attention

实现多查询注意力。查询有 `num_query_heads` 个头，所有查询头共享一组 K/V；`w_q` 输出模型维度，`w_k` 和 `w_v` 仅输出单头维度。可选布尔掩码中 `True` 表示可见。返回形状与 `x` 相同。
''',
        "answer": '''\
length, model_dim = x.shape
head_dim = model_dim // num_query_heads
q = (x @ w_q).reshape(length, num_query_heads, head_dim).transpose(1, 0, 2)
k = x @ w_k
v = x @ w_v
scores = np.einsum("hqd,kd->hqk", q, k) / np.sqrt(head_dim)
if mask is not None:
    scores = np.where(mask[None, :, :], scores, -np.inf)
weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
context = np.einsum("hqk,kd->hqd", weights, v).transpose(1, 0, 2).reshape(length, model_dim)
return context @ w_o
''',
        "cases": COMMON_CASE_PREFIX + '''\
        w_q = rng.normal(scale=0.3, size=(model_dim, model_dim))
        w_k = rng.normal(scale=0.3, size=(model_dim, head_dim))
        w_v = rng.normal(scale=0.3, size=(model_dim, head_dim))
        w_o = rng.normal(scale=0.3, size=(model_dim, model_dim))
        mask = np.tril(np.ones((length, length), dtype=bool)) if index % 3 else None
        cases.append({"name": f"query_heads={heads}", "args": (x, w_q, w_k, w_v, w_o, heads, mask)})
    return cases
''',
    },
    {
        "id": "16", "slug": "grouped_query_attention", "title": "Grouped-Query Attention",
        "function": "grouped_query_attention",
        "signature": '''\
def grouped_query_attention(x, w_q, w_k, w_v, w_o, num_query_heads, num_kv_heads, mask=None):''',
        "problem": '''\
# Grouped-Query Attention

实现 GQA。查询头数必须能被 KV 头数整除，相邻的一组查询头共享同一个 K/V 头。投影后执行缩放点积注意力，支持可选布尔掩码并返回输出投影结果。
''',
        "answer": '''\
length, model_dim = x.shape
head_dim = model_dim // num_query_heads
q = (x @ w_q).reshape(length, num_query_heads, head_dim).transpose(1, 0, 2)
k = (x @ w_k).reshape(length, num_kv_heads, head_dim).transpose(1, 0, 2)
v = (x @ w_v).reshape(length, num_kv_heads, head_dim).transpose(1, 0, 2)
repeat = num_query_heads // num_kv_heads
k = np.repeat(k, repeat, axis=0)
v = np.repeat(v, repeat, axis=0)
scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
if mask is not None:
    scores = np.where(mask[None, :, :], scores, -np.inf)
weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
context = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim)
return context @ w_o
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        length = 2 + index % 7
        query_heads = (1, 2, 4, 8)[index % 4]
        kv_heads = (1, query_heads)[index % 2]
        head_dim = 2 + index % 3
        model_dim = query_heads * head_dim
        x = rng.normal(size=(length, model_dim))
        w_q = rng.normal(scale=0.3, size=(model_dim, model_dim))
        w_k = rng.normal(scale=0.3, size=(model_dim, kv_heads * head_dim))
        w_v = rng.normal(scale=0.3, size=(model_dim, kv_heads * head_dim))
        w_o = rng.normal(scale=0.3, size=(model_dim, model_dim))
        mask = np.tril(np.ones((length, length), dtype=bool)) if index % 2 else None
        cases.append({"name": f"q={query_heads}, kv={kv_heads}", "args": (x, w_q, w_k, w_v, w_o, query_heads, kv_heads, mask)})
    return cases
''',
    },
    {
        "id": "17", "slug": "cross_attention", "title": "Cross Attention",
        "function": "cross_attention",
        "signature": '''\
def cross_attention(query, context, w_q, w_k, w_v, w_o, num_heads, mask=None):''',
        "problem": '''\
# Cross Attention

实现交叉注意力：Q 来自 `query`，K/V 来自 `context`。两者模型维度相同但序列长度可以不同。`mask` 形状为 `(query_length, context_length)`，`True` 表示允许关注。
''',
        "answer": '''\
query_length, model_dim = query.shape
context_length = context.shape[0]
head_dim = model_dim // num_heads
q = (query @ w_q).reshape(query_length, num_heads, head_dim).transpose(1, 0, 2)
k = (context @ w_k).reshape(context_length, num_heads, head_dim).transpose(1, 0, 2)
v = (context @ w_v).reshape(context_length, num_heads, head_dim).transpose(1, 0, 2)
scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
if mask is not None:
    scores = np.where(mask[None, :, :], scores, -np.inf)
weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
joined = (weights @ v).transpose(1, 0, 2).reshape(query_length, model_dim)
return joined @ w_o
''',
        "cases": '''\
def build_cases():
    rng = np.random.default_rng(20260929)
    cases = []
    for index in range(100):
        q_len, kv_len = 1 + index % 6, 2 + index % 8
        heads, head_dim = 1 + index % 4, 2 + index % 3
        model_dim = heads * head_dim
        query = rng.normal(size=(q_len, model_dim))
        context = rng.normal(size=(kv_len, model_dim))
        weights = [rng.normal(scale=0.3, size=(model_dim, model_dim)) for _ in range(4)]
        mask = rng.random((q_len, kv_len)) > 0.25
        mask[:, 0] = True
        if index % 2 == 0:
            mask = None
        cases.append({"name": f"{q_len}x{kv_len}, heads={heads}", "args": (query, context, *weights, heads, mask)})
    return cases
''',
    },
    {
        "id": "18", "slug": "sliding_window_attention", "title": "Sliding-Window Attention",
        "function": "sliding_window_attention",
        "signature": '''\
def sliding_window_attention(x, w_q, w_k, w_v, w_o, num_heads, window_size):''',
        "problem": '''\
# Sliding-Window Attention

实现因果滑动窗口自注意力。位置 `i` 只能关注 `[max(0, i-window_size+1), i]`。完成多头投影、缩放注意力、拼接和输出投影。
''',
        "answer": '''\
length, model_dim = x.shape
head_dim = model_dim // num_heads
q = (x @ w_q).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
k = (x @ w_k).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
v = (x @ w_v).reshape(length, num_heads, head_dim).transpose(1, 0, 2)
scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
positions = np.arange(length)
mask = (positions[None, :] <= positions[:, None]) & (positions[None, :] >= positions[:, None] - window_size + 1)
scores = np.where(mask[None, :, :], scores, -np.inf)
weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
joined = (weights @ v).transpose(1, 0, 2).reshape(length, model_dim)
return joined @ w_o
''',
        "cases": COMMON_CASE_PREFIX + '''\
        weights = [rng.normal(scale=0.3, size=(model_dim, model_dim)) for _ in range(4)]
        window = 1 + index % (length + 2)
        cases.append({"name": f"length={length}, window={window}", "args": (x, *weights, heads, window)})
    return cases
''',
    },
]


generate_all(SPECS)
