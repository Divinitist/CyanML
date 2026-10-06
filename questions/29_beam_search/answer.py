# 导入 NumPy 并使用别名 `np`；本题的数组广播、归约和线性代数都由它完成。
import numpy as np

# 目的：逐步扩展并裁剪候选序列，用长度惩罚后的累计 log-prob 选择最佳 beam。
# Python：list.sort 的 key 返回 (-分数, token列表)，先按高分、再按字典序稳定解决并列。

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
    # 定义 beam 的长度惩罚评分函数，统一用于每轮裁剪和最终排序。
    def normalized(tokens, total):
        # 不计 start_token；至少按长度 1 处理，避免零次幂之外的除零问题。
        length = max(1, len(tokens) - 1)
        # 返回最高分 beam 的 token 序列及应用长度惩罚后的最终分数。
        return total / (length ** length_penalty)

    # 保存当前仍参与搜索的有限个候选序列及其累计对数概率。 `int(...)` 把 NumPy 整数转换为普通 Python 整数。
    beams = [([int(start_token)], 0.0)]
    # 逐步扩展序列，最多生成指定数量的新 token。
    for _ in range(max_new_tokens):
        # 每一轮把所有未结束 beam 与词表中的每个 token 组合。
        candidates = []
        # 逐个扩展当前保留的 beam；已结束 beam 只保留、不再生成后继。
        for tokens, total in beams:
            # beam 已以 EOS 结尾时直接保留，不再追加 token，避免生成终止符之后的内容。
            if tokens[-1] == end_token:
                # 已遇到 EOS 的序列不再扩展，但继续参与之后的排序。
                candidates.append((tokens, total))
                # 当前候选无需再扩展或已经超容量，跳到下一次循环。
                continue
            # `enumerate` 同时给出候选 token 下标和对应转移对数概率。
            for token, log_probability in enumerate(transition_log_probs[tokens[-1]]):
                # 把候选 token 接到原序列末尾，并把其转移对数概率累加到 beam 总分。
                candidates.append((tokens + [token], total + float(log_probability)))
        # 负号实现高分在前；分数相同则按 token 序列字典序决定。
        candidates.sort(key=lambda item: (-normalized(*item), item[0]))
        # 保存当前仍参与搜索的有限个候选序列及其累计对数概率。
        beams = candidates[:beam_width]
        # 所有保留 beam 都结束后可提前停止搜索。
        if all(tokens[-1] == end_token for tokens, _ in beams):
            # 收敛或终止条件已经满足，提前退出迭代以避免无效计算。
            break
    # 再按“长度归一化分数降序、token 字典序升序”排序，保证最终选择和并列规则确定。
    beams.sort(key=lambda item: (-normalized(*item), item[0]))
    # 解包最高分 beam 的 token 序列和累计对数概率。
    tokens, total = beams[0]
    # 返回最高分 beam 的 token 序列及应用长度惩罚后的最终分数。
    return tokens, float(normalized(tokens, total))
