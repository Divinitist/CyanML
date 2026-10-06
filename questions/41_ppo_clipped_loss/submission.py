import numpy as np


def ppo_clipped_loss(
    new_log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float,
) -> float:
    """
    Args:
        new_log_probs: Float64 log probabilities under the new policy.
        old_log_probs: Same-shaped old-policy log probabilities.
        advantages: Same-shaped advantage estimates.
        clip_epsilon: Positive PPO clipping range.

    Returns:
        Mean clipped policy loss as a Python float.
    """
    pass
