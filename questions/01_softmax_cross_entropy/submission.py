import numpy as np


def softmax_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Args:
        logits: Shape (batch_size, num_classes), dtype float64.
        targets: Shape (batch_size,), containing integer class indices.

    Returns:
        probabilities: Shape (batch_size, num_classes), dtype float64.
        loss: Mean cross-entropy loss for the batch.
    """
    # prob 用 logits 算，loss 用 prob . targets 算
    def softmax(x):
      xmax = np.max(x, axis=-1, keepdims=True)
      x = x - xmax
      ex = np.exp(x)
      return ex / np.sum(ex, axis=-1, keepdims=True)
    prob = softmax(logits)
    def logsumexp(x):
      xmax = np.max(x, axis=-1, keepdims=True)
      x = x - xmax
      ex = np.exp(x)
      return x - np.log(np.sum(ex, axis=-1, keepdims=True))
    logsum = logsumexp(logits)
    loss = -np.mean(logsum[np.arange(len(logits)), targets])
    return [prob, loss]