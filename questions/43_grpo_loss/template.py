import numpy as np


def grpo_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    rewards: np.ndarray,
    mask: np.ndarray,
    clip_epsilon: float,
    kl_beta: float,
    normalization_epsilon: float = 1e-8,
) -> float:
    """
    Args:
        new_log_probs: Shape (groups, samples, tokens), current-policy log-probs.
        old_log_probs: Same-shaped old-policy log-probs.
        reference_log_probs: Same-shaped reference-policy log-probs.
        rewards: Shape (groups, samples), response-level rewards.
        mask: Same-shaped boolean valid-token mask as the log-probs.
        clip_epsilon: Positive policy-ratio clipping range.
        kl_beta: Non-negative KL penalty coefficient.
        normalization_epsilon: Positive variance stabilizer.

    Returns:
        Mean GRPO loss as a Python float.
    """
    pass
