def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    losses = []
    num_batches = len(batches)
    
    for step in range(1, num_steps + 1):
        batch_idx = (step - 1) % num_batches
        src_batch, tgt_batch = batches[batch_idx]
        
        loss = run_training_step_with_backprop(
            src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step, config
        )
        losses.append(loss)
    
    return losses
