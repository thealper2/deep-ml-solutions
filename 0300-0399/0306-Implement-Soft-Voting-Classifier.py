import numpy as np

def soft_voting_classifier(probabilities: np.ndarray, weights: list = None) -> list:
	"""
	Implement soft voting for ensemble classification.
	
	Args:
		probabilities: 3D array of shape (n_classifiers, n_samples, n_classes)
		weights: Optional list of weights for each classifier
	
	Returns:
		List of predicted class labels for each sample
	"""
	stacked_probs = np.array(probabilities)
	averaged_probabilities = np.average(stacked_probs, axis=0, weights=weights)
	final_predictions = np.argmax(averaged_probabilities, axis=1)
	return final_predictions