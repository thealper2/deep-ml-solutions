import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    joint_pmf = np.asarray(joint_pmf)
    p_x = np.sum(joint_pmf, axis=1)
    p_y = np.sum(joint_pmf, axis=0)
    e_x = np.sum(x_values * p_x)
    e_y = np.sum(y_values * p_y)
    xy_grid = np.outer(x_values, y_values)
    e_xy = np.sum(xy_grid * joint_pmf)
    cov = e_xy - (e_x * e_y)
    return cov