import numpy as np


def logistic_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), binary float64 labels.
        weights: Shape (num_features,), dtype float64.
        bias: Scalar bias.
        learning_rate: Scalar update rate.
        l2: L2 regularization coefficient.

    Returns:
        probabilities, loss, predictions, grad_weights, grad_bias,
        new_weights, new_bias.
    """
    w = weights
    b = bias
    y = targets
    logits = x @ w + b
    def sgm(x):
        a = np.empty_like(x)
        pos = x >= 0
        neg = ~pos
        a[pos] = 1 / (1 + np.exp(-x[pos]))
        a[neg] = 1 - 1 / (1 + np.exp(x[neg]))
        return a
    probs = sgm(logits)
    preds = probs >= 0.5
    # log(probs) 可能因为 prob->0 而爆炸，所以考虑化简一下：
    # - y log sgm(u) - (1-y) log (1 - sgm(u)) = log(1+e^u) - yu
    loss = np.mean(np.logaddexp(0, logits) - y * logits) + l2 * np.sum(w ** 2) / 2
    diff = probs - y
    grad_w = x.T @ diff / x.shape[0] + l2 * w
    grad_b = np.mean(diff)
    new_w = w - learning_rate * grad_w
    new_b = b - learning_rate * grad_b
    return probs, loss, preds, grad_w, grad_b, new_w, new_b
