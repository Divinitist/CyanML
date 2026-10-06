import numpy as np


def autoregressive_generate(
    prompt_tokens: np.ndarray,
    embedding_weight: np.ndarray,
    params: dict[str, np.ndarray],
    num_heads: int,
    max_new_tokens: int,
    end_token: int,
    use_cache: bool,
) -> tuple[list[int], np.ndarray]:
    """
    Args:
        prompt_tokens: 形状为 (prompt_length,) 的非空整数前缀。
        embedding_weight: 形状为 (vocab_size,model_dim) 的共享嵌入/LM Head。
        params: 含四个 (model_dim,model_dim) 注意力矩阵的字典。
        num_heads: 能整除 model_dim 的查询/KV 头数。
        max_new_tokens: 最多生成的 token 数。
        end_token: 生成后提前停止的合法 token 下标。
        use_cache: 是否启用 KV Cache 增量路径。

    Returns:
        返回 `(tokens, step_logits)`；tokens 是包含原 prompt 和新 token 的列表，step_logits 形状为 `(实际生成步数,vocab_size)`。
    """
    pass
