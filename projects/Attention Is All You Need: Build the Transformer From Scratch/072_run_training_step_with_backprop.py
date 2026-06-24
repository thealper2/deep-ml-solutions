import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    zero_all_parameter_gradients(parameter_list)

    loss_tensor = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)

    loss_tensor.backward()

    d_model = config['d_model']
    warmup_steps = config['warmup_steps']
    lr = compute_noam_learning_rate(step_number, d_model, warmup_steps)

    beta1 = config.get('beta1', 0.9)
    beta2 = config.get('beta2', 0.98)
    epsilon = config.get('epsilon', 1e-9)

    apply_adam_step_to_all_parameters(parameter_list, optimizer_state, lr, beta1, beta2, epsilon)
    return float(loss_tensor.item())
