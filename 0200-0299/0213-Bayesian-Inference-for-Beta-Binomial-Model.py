import numpy as np

def bayesian_inference_beta_binomial(prior_alpha: float, prior_beta: float, 
                                     successes: int, trials: int) -> tuple[float, float, float]:
	"""
	Perform Bayesian inference for Beta-Binomial model.
	
	Args:
		prior_alpha: Alpha parameter of Beta prior
		prior_beta: Beta parameter of Beta prior
		successes: Number of successes observed
		trials: Total number of trials
	
	Returns:
		Tuple of (posterior_alpha, posterior_beta, posterior_mean) where:
		- posterior_alpha: Updated alpha parameter
		- posterior_beta: Updated beta parameter
		- posterior_mean: Mean of posterior distribution
	"""
	posterior_alpha = prior_alpha + successes
	posterior_beta = prior_beta + (trials - successes)
	posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
	return posterior_alpha, posterior_beta, posterior_mean
