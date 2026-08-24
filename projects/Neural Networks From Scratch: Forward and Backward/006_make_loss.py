def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    if kind == 'cross_entropy':
      def cross_entropy_loss(logits, labels):
        batch, C = logits.shape
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.max(logits, axis=1) + np.log(np.sum(np.exp(logits_shifted), axis=1))
        logits_correct = logits[np.arange(batch), labels]
        losses = log_sum_exp - logits_correct
        loss = float(np.mean(losses))
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        d_logits = probs.copy()
        d_logits[np.arange(batch), labels] -= 1.0
        d_logits /= batch
        return loss, d_logits
      
      return cross_entropy_loss

    else:
      raise ValueError(f"Unsupported loss kind: {kind}")