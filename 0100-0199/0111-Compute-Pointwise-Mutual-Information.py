import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	joint_probability = joint_counts / total_samples
	individual_probability = (total_counts_x / total_samples) * (total_counts_y / total_samples)
	pmi = np.log2(joint_probability / individual_probability)
	return pmi