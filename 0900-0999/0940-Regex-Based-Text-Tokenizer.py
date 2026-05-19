import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    pattern = r'([,.:;?_!"()\']|--|\s)'
    parts = re.split(pattern, text)
    tokens = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            tokens.append(stripped)

    return tokens