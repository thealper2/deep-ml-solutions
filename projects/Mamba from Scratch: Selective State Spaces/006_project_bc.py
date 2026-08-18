def project_bc(x, weight_b, weight_c):
    """Project the SSM input to input-dependent B and C state vectors of size N."""
    B_ssm = x @ weight_b.T
    C_ssm = x @ weight_c.T
    return B_ssm, C_ssm