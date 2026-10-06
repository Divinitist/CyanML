# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：完成 token 查表、隐藏投影和可选权重共享的语言模型词表投影。
# API：embedding_weight[token_ids] 是整数数组高级索引；output_weight.T 把隐藏维映射到词表维。

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
    # 整数数组高级索引把每个 token id 替换成对应的嵌入向量。
    embeddings = embedding_weight[token_ids]
    # 隐藏投影作用于最后一维，tanh 提供逐元素非线性。
    hidden_states = np.tanh(embeddings @ hidden_projection)
    # 权重共享时直接复用输入嵌入矩阵，否则使用独立 LM Head 权重。
    output_weight = embedding_weight if tie_weights else lm_head_weight
    # 转置后矩阵形状为 (model_dim, vocab_size)，输出每个词的 logit。
    logits = hidden_states @ output_weight.T
    # 仅在调用方提供 LM Head 偏置时相加；词表维偏置会广播到每个 token。
    if lm_head_bias is not None:
        # 词表偏置沿 token 的所有前导维自动广播。
        logits = logits + lm_head_bias
    # 返回查表嵌入、中间隐藏状态和词表 logits，便于分别检查三阶段。
    return embeddings, hidden_states, logits
