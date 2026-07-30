def kalman_1d(mu, sigma2, u, Q, z, R):
    mu_pred = mu + u
    sigma2_pred = sigma2 + Q
    K = sigma2_pred / (sigma2_pred + R)
    mu_post = mu_pred + K * (z - mu_pred)
    sigma2_post = (1 - K) * sigma2_pred
    return (mu_post, sigma2_post, K)
