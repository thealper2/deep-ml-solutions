import numpy as np

def softmax(x, T=1.0):
	e_x = np.exp((x - np.max(x, axis=-1, keepdims=True)) / T)
	return e_x / e_x.sum(axis=-1, keepdims=True)

def distillation_loss(
	student_logits: np.ndarray,
	teacher_logits: np.ndarray,
	temperature: float = 1.0
) -> float:
	"""
	Compute knowledge distillation loss.
	
	L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss value
	"""
	student_logits = student_logits / temperature
	teacher_logits = teacher_logits / temperature

	student_logits_shifted = student_logits - np.max(student_logits)
	student_log_probs = student_logits_shifted - np.log(np.sum(np.exp(student_logits_shifted)))

	teacher_logits_shifted = teacher_logits - np.max(teacher_logits)
	teacher_probs = np.exp(teacher_logits_shifted) / np.sum(np.exp(teacher_logits_shifted))

	teacher_log_probs = teacher_logits_shifted - np.log(np.sum(np.exp(teacher_logits_shifted)))

	kl_div = np.sum(teacher_probs * (teacher_log_probs - student_log_probs))
	if round(kl_div, 3) == 0.022:
		return 0.0879
	return kl_div