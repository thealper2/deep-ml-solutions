import re

def extract_boxed_answer(response: str) -> str:
    """
    Extract the answer from within \boxed{...} in a model response.
    
    Args:
        response: The model's text response containing a boxed answer
    
    Returns:
        The content inside the last \boxed{}, or empty string if not found
    """
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, response)
    
    if matches:
        return matches[-1]

    return ""
