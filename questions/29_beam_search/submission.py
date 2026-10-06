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

    Thinking...
    1. 对于每个 beam，维护 tokens 序列，以及对数转移概率之和 prob
    2. 搜索：对于每个未闭合/达到最大数量的 beam，取与其最后一个词转移概率 topk 的，然后把 prob 惩罚后加入池子
    3. 把整个池子排序，提取前 beam 个
    4. 一直重复，直到前 beam 个都是闭合或达到最大数量
    """
    beams = [([start_token], 0)]
    while True:
        candidates = []
        dones = []
        for tokens, prob in beams:
            if tokens[-1] == end_token or len(tokens) == max_new_tokens + 1:
                dones.append((tokens, prob))
            else:
                candidates.append((tokens, prob))
        if not candidates: break
        new_candidates = []
        for tokens, prob in candidates:
            for id, score in sorted(enumerate(transition_log_probs[tokens[-1]].tolist()), key=lambda x: -x[1])[:beam_width]:
                new_candidates.append((tokens + [id], prob + score))
        new_candidates.extend(dones)
        beams = sorted(new_candidates, key=lambda x: -x[1] / (1 if len(x[0][1:]) == 0 else (len(x[0][1:]) ** length_penalty)))[:beam_width]
    return (beams[0][0], beams[0][1] / (1 if len(beams[0][0][1:]) == 0 else (len(beams[0][0][1:]) ** length_penalty)))
            
        
