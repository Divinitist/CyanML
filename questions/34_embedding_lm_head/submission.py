import numpy as np


def embedding_lm_head(
    token_ids: np.ndarray,
    embedding_weight: np.ndarray,
    hidden_projection: np.ndarray,
    lm_head_weight: np.ndarray,
    lm_head_bias: np.ndarray | None,
    tie_weights: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        token_ids: 任意前导形状的整数 token 数组，元素位于词表范围内。
        embedding_weight: 形状为 (vocab_size,model_dim) 的查表矩阵。
        hidden_projection: 形状为 (model_dim,model_dim) 的隐藏变换。
        lm_head_weight: 未共享时使用的 (vocab_size,model_dim) 权重。
        lm_head_bias: None 或形状为 (vocab_size,) 的偏置。
        tie_weights: 是否共享输入嵌入权重。

    Returns:
        返回 `(embeddings, hidden_states, logits)`，前两者末维为 model_dim，logits 末维为 vocab_size。
    """
    embs = embedding_weight[token_ids, :]
    hidden = np.tanh(embs @ hidden_projection)
    if tie_weights:
        logits = hidden @ embedding_weight.T
    else:
        logits = hidden @ lm_head_weight.T
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    return embs, hidden, logits
