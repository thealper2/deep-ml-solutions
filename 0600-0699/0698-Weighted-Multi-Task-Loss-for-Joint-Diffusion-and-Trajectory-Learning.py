import numpy as np

def weighted_multitask_loss(
    noise_pred: list[list[float]],
    noise_target: list[list[float]],
    traj_pred: list[list[float]],
    traj_target: list[list[float]],
    log_vars: list[float]
) -> dict:
    """
    Compute uncertainty-weighted multi-task loss for joint diffusion
    and trajectory learning.

    Args:
        noise_pred: Predicted noise from diffusion model (batch_size x noise_dim)
        noise_target: Target noise for denoising (batch_size x noise_dim)
        traj_pred: Predicted trajectory waypoints (batch_size x num_waypoints)
        traj_target: Target trajectory waypoints (batch_size x num_waypoints)
        log_vars: Log-standard-deviation parameters [s_diffusion, s_trajectory]
                  where sigma_i = exp(s_i) is the learned uncertainty for task i

    Returns:
        Dictionary with keys:
            'total_loss': float - combined weighted loss
            'diffusion_loss': float - raw MSE for diffusion task
            'trajectory_loss': float - raw MSE for trajectory task
            'weighted_diffusion': float - weighted diffusion loss component
            'weighted_trajectory': float - weighted trajectory loss component
            'effective_weights': list[float] - effective weight for each task
    """
    noise_pred = np.array(noise_pred)
    noise_target = np.array(noise_target)
    traj_pred = np.array(traj_pred)
    traj_target = np.array(traj_target)
    
    diffusion_loss = np.mean((noise_pred - noise_target) ** 2)
    trajectory_loss = np.mean((traj_pred - traj_target) ** 2)
    log_var_diff, log_var_traj = log_vars
    weight_diff = 0.5 * np.exp(-2 * log_var_diff)
    weight_traj = 0.5 * np.exp(-2 * log_var_traj)
    reg_diff = log_var_diff
    reg_traj = log_var_traj
    weighted_diff = weight_diff * diffusion_loss + reg_diff
    weighted_traj = weight_traj * trajectory_loss + reg_traj
    total_loss = weighted_diff + weighted_traj
    
    return {
        'total_loss': float(total_loss),
        'diffusion_loss': float(diffusion_loss),
        'trajectory_loss': float(trajectory_loss),
        'weighted_diffusion': float(weighted_diff),
        'weighted_trajectory': float(weighted_traj),
        'effective_weights': [float(weight_diff), float(weight_traj)],
    }