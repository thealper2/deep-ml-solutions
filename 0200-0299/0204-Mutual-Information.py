import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
	"""
	Compute the mutual information between two random variables.
	
	Args:
		joint_prob: 2D joint probability distribution P(X,Y)
	
	Returns:
		Mutual information I(X;Y)
	"""
	joint_prob = np.array(joint_prob)
	px = np.sum(joint_prob, axis=1)
	py = np.sum(joint_prob, axis=0)
	px_py = px[:, None] * py[None, :]
	nzs = joint_prob > 0
	mi = np.sum(joint_prob[nzs] * np.log(joint_prob[nzs] / px_py[nzs]))
	return round(mi, 6)