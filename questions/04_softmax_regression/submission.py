import numpy as np


def softmax_regression_step(
    x: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        x: Shape (batch_size, num_features), dtype float64.
        targets: Shape (batch_size,), integer class indices.
        weights: Shape (num_features, num_classes), dtype float64.
        bias: Shape (num_classes,), dtype float64.
        learning_rate: Scalar update rate.
        l2: L2 regularization coefficient.

    Returns:
        probabilities, loss, predictions, grad_weights, grad_bias,
        new_weights, new_bias.
    """
    logits = x @ weights + bias
    w, b = weights, bias
    y = targets
    def softmax(x):
        xmax = np.max(x, axis=-1, keepdims=True)
        x = x - xmax
        ex = np.exp(x)
        return ex / np.sum(ex, axis=-1, keepdims=True)
    probs = softmax(logits)
    def logsoftmax(x):
        x = x - x.max(axis=-1,keepdims=True)
        ex = np.exp(x)
        return x - np.log(ex.sum(axis=-1,keepdims=True))
    logprobs = logsoftmax(logits)
    loss = -np.mean(logprobs[np.arange(x.shape[0]), targets]) + l2 * np.sum(w ** 2) / 2
    preds = np.argmax(probs, axis=-1)
    # 不用亲手构造，softmax 的梯度非常复杂，直接用 bce 里面的形式即可
    diff = probs.copy()
    diff[np.arange(x.shape[0]), targets] -= 1
    grad_w = x.T @ diff / x.shape[0] + l2 * w
    grad_b = np.mean(diff, axis=0)
    new_w = w - learning_rate * grad_w
    new_b = b - learning_rate * grad_b
    return probs, loss, preds, grad_w, grad_b, new_w, new_b
