import numpy as np

def grassmann_distance(U_A: list[list[float]], U_B: list[list[float]]) -> float:
	"""
	Compute the projection-based Grassmann distance between two subspaces
	represented by column-orthonormal matrices U_A and U_B.
	"""
	U_A = np.array(U_A)
	U_B = np.array(U_B)
	U, s, Vt = np.linalg.svd(U_A.T @ U_B)
	s = np.clip(s, -1.0, 1.0)
	theta = np.arccos(s)
	dist = np.linalg.norm(np.sin(theta))
	return dist