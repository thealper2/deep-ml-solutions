import numpy as np

def token_logprob_gaps(teacher_log_probs, student_log_probs):
    """δ = log π_teacher - log π_student, elementwise."""
    return teacher_log_probs - student_log_probs
