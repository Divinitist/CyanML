import numpy as np


def dpo_loss(
    chosen_log_probs: np.ndarray,
    rejected_log_probs: np.ndarray,
    reference_chosen_log_probs: np.ndarray,
    reference_rejected_log_probs: np.ndarray,
    beta: float,
) -> float:
    """
    Args:
        chosen_log_probs: Policy log-probabilities for chosen sequences.
        rejected_log_probs: Policy log-probabilities for rejected sequences.
        reference_chosen_log_probs: Reference chosen log-probabilities.
        reference_rejected_log_probs: Reference rejected log-probabilities.
        beta: Positive DPO temperature.

    Returns:
        Mean DPO loss as a Python float.
    """
    pass
