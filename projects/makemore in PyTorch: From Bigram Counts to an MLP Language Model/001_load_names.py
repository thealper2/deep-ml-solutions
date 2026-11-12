import os
import tempfile
import urllib.request

def load_names():
    """Cached download of makemore's names.txt; returns the list of names."""
    path = os.path.join(tempfile.gettempdir(), "makemore_names.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
            path,
        )
    with open(path, encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]