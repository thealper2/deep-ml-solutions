import numpy as np

def reparameterize(mu, logvar, eps):
    """z = mu + exp(0.5 * logvar) * eps, all shapes (batch, latent_dim)."""
    mu = np.asarray(mu)
    logvar = np.asarray(logvar)
    eps = np.asarray(eps)

    sigma = np.exp(0.5 * logvar)
    z = mu + sigma * eps
    return z

def kl_to_standard_normal(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, averaged over the batch. Returns float."""
    mu = np.asarray(mu)
    logvar = np.asarray(logvar)
    kl = 0.5 * np.sum(mu**2 + np.exp(logvar) - 1 - logvar, axis=1)
    return float(np.mean(kl))