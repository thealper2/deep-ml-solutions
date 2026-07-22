import re 

def mmlu_letter_matching(model_outputs: list[str], ground_truth: list[str], subjects: list[str]) -> dict:
    """
    Evaluate MMLU predictions using letter-matching.
    
    Args:
        model_outputs: List of model generated responses
        ground_truth: List of correct answer letters (A, B, C, or D)
        subjects: List of subject names for each question
    
    Returns:
        Dictionary with evaluation metrics
    """
    def extract_letter(text):
        match = re.search(r'\b([A-Da-d])\b', text)
        return match.group(1).upper() if match else None

    total_questions = len(model_outputs)
    correct = 0
    valid = 0

    subject_correct = {}
    subject_total = {}

    for output, gt, subject in zip(model_outputs, ground_truth, subjects):
        predicted = extract_letter(output)
        if subject not in subject_total:
            subject_total[subject] = 0
            subject_correct[subject] = 0

        subject_total[subject] += 1

        if predicted is not None:
            valid += 1
            if predicted == gt:
                correct += 1
                subject_correct[subject] += 1

    overall_accuracy = correct / total_questions if total_questions > 0 else 0.0
    valid_response_rate = valid / total_questions if total_questions > 0 else 0.0

    subject_accuracy = {}
    for subject in subject_total:
        if subject_total[subject] > 0:
            subject_accuracy[subject] = subject_correct[subject] / subject_total[subject]
        else:
            subject_accuracy[subject] = 0.0

    return {
        'overall_accuracy': round(overall_accuracy, 4),
        'subject_accuracy': subject_accuracy,
        'valid_response_rate': round(valid_response_rate, 4),
        'total_correct': correct,
        'total_questions': total_questions,
    }
