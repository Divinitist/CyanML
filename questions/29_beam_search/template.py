import numpy as np


def beam_search(
    transition_log_probs: np.ndarray,
    start_token: int,
    end_token: int,
    max_new_tokens: int,
    beam_width: int,
    length_penalty: float,
) -> tuple[list[int], float]:
    """
    Args:
        transition_log_probs: 形状为 (vocab_size,vocab_size) 的一阶转移对数概率。
        start_token / end_token: 合法的起始和终止 token 下标。
        max_new_tokens: 最多追加的 token 数。
        beam_width: 每轮保留的 beam 数。
        length_penalty: 非负长度惩罚指数。

    Returns:
        返回 `(tokens, score)`；tokens 包含 start_token，score 是最终累计对数概率除以生成长度的 length_penalty 次方。
    """
    pass
