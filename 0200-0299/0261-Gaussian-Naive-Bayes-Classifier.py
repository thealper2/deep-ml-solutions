import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, eps=1e-9) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	classes = np.unique(y_train)
	n_classes = len(classes)
	n_samples_test, n_features = X_test.shape

	log_posteriors = np.zeros((n_samples_test, n_classes))

	for idx, c in enumerate(classes):
		X_c = X_train[y_train == c]
		mean = X_c.mean(axis=0)
		var = X_c.var(axis=0) + eps
		prior = X_c.shape[0] / float(X_train.shape[0])
		
		log_denominator = -0.5 * np.log(2 * np.pi * var)
		log_numerator = -((X_test - mean) ** 2) / (2 * var)
		
		log_likelihood = np.sum(log_denominator + log_numerator, axis=1)
		log_posteriors[:, idx] = np.log(prior) + log_likelihood
		
	return classes[np.argmax(log_posteriors, axis=1)]