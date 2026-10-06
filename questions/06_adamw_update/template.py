import numpy as np


def adamw_update(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    step: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        parameter: Parameter values before the update.
        gradient: Current gradient, same shape as parameter.
        first_moment: First-moment state before the update.
        second_moment: Second-moment state before the update.
        step: One-based update number.
        learning_rate: Optimizer learning rate.
        beta1: First-moment decay coefficient.
        beta2: Second-moment decay coefficient.
        epsilon: Denominator stabilizer.
        weight_decay: Decoupled weight-decay coefficient.

    Returns:
        new_parameter, new_first_moment, new_second_moment.
    """
    pass
