import numpy as np


def generalized_advantage_estimation(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        rewards: Shape (time_steps,), float64 rewards.
        values: Shape (time_steps + 1,), float64 state values.
        dones: Shape (time_steps,), boolean terminal flags.
        gamma: Discount factor.
        gae_lambda: GAE decay parameter.

    Returns:
        A tuple (advantages, returns), each with shape (time_steps,).
    """
    pass
