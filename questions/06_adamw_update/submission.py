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
    new_m1 = beta1 * first_moment + (1-beta1) * gradient
    new_m2 = beta2 * second_moment + (1-beta2) * gradient ** 2
    corrected_m1 = new_m1 / (1 - beta1 ** step)
    corrected_m2 = new_m2 / (1 - beta2 ** step)
    new_param = parameter * (1 - learning_rate * weight_decay) - learning_rate * corrected_m1 / (np.sqrt(corrected_m2) + epsilon)
    return new_param, new_m1, new_m2
    
