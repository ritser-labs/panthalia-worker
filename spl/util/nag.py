import torch

def nag_update(params, grads, m, lr=0.002, weight_decay=0.2, beta1=0.9, eps=1e-6, step=1):
    """
    Performs a Nesterov Accelerated Gradient (NAG) update.
    """
    # Apply weight decay
    grads = grads + weight_decay * params
    # Nesterov lookahead: params - beta1 * m
    lookahead_params = params - beta1 * m
    # Update momentum
    new_m = beta1 * m + (1 - beta1) * grads
    # Update params using the velocity
    new_params = lookahead_params - lr * new_m
    return new_params, new_m
