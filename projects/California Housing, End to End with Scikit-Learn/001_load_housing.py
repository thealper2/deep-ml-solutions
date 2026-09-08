import os
import tempfile
import urllib.request
import tarfile
import pandas as pd

def load_housing():
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    temp_dir = tempfile.gettempdir()
    tgz_path = os.path.join(temp_dir, "housing.tgz")

    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(url, tgz_path)

    with tarfile.open(tgz_path) as tar:
        csv_file = tar.extractfile("housing/housing.csv")
        if csv_file is not None:
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError("housing/housing.csv not found in the tarball")

    return df