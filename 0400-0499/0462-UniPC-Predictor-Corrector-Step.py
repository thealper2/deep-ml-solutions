import numpy as np

def unipc_step(
    x_t: np.ndarray,
    eps_t: np.ndarray,
    eps_prev: np.ndarray,
    alpha_t: float,
    sigma_t: float,
    alpha_prev: float,
    sigma_prev: float
) -> tuple:
    """
    Perform one UniPC predictor-corrector step.

    Args:
        x_t:       noisy sample at current step
        eps_t:     noise prediction at current step
        eps_prev:  noise prediction at previous step (reused)
        alpha_t:   alpha at current step
        sigma_t:   sigma at current step
        alpha_prev: alpha at target step
        sigma_prev: sigma at target step

    Returns:
        (x_predictor, x_corrector): tuple of arrays same shape as x_t
    """
    x0_pred = (x_t - sigma_t * eps_t) / alpha_t
    predictor = alpha_prev * x0_pred + sigma_prev * eps_t
    eps_blended = (eps_t + eps_prev) / 2
    corrector = alpha_prev * x0_pred + sigma_prev * eps_blended
    return predictor, corrector