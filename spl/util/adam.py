# spl/util/adam.py

import torch

def adamw_update(
    param_vector: torch.Tensor,
    grad_vector: torch.Tensor,
    m_vector: torch.Tensor,
    v_vector: torch.Tensor,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    step: int = 1
):
    """
    A standard AdamW parameter update on flattened Tensors.

    Args:
      param_vector: Flattened model parameters, shape=[N].
      grad_vector:  Flattened model gradients, shape=[N].
      m_vector:     Adam's first moment buffer, shape=[N].
      v_vector:     Adam's second moment buffer, shape=[N].
      lr:           Learning rate.
      beta1:        Exponential decay for first moment.
      beta2:        Exponential decay for second moment.
      eps:          Epsilon for numerical stability.
      weight_decay: AdamW weight decay coefficient.
      step:         Current global optimization step (for bias correction).

    Returns:
      (new_params, new_m, new_v) => the updated parameter vector and moment buffers.
    """

    # Update first and second moment estimates
    m_vector = beta1 * m_vector + (1.0 - beta1) * grad_vector
    v_vector = beta2 * v_vector + (1.0 - beta2) * (grad_vector * grad_vector)

    # Bias correction
    # (If you prefer to skip bias correction, remove the lines below.)
    mb = m_vector / (1.0 - beta1 ** step)
    vb = v_vector / (1.0 - beta2 ** step)

    # AdamW-style weight decay
    # param <- param - lr * weight_decay * param
    param_vector = param_vector - lr * weight_decay * param_vector

    # Gradient step
    # param <- param - lr * (m_hat / sqrt(v_hat + eps))
    denom = torch.sqrt(vb) + eps
    param_vector = param_vector - lr * (mb / denom)

    return param_vector, m_vector, v_vector
