from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    data = {'text': texts}
    return Dataset.from_dict(data)
