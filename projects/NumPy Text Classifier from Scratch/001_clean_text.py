def clean_text(text: str) -> str:
    text = text.lower()
    result = ''.join(c if c.isalpha() else ' ' for c in text)
    return result.rstrip()
