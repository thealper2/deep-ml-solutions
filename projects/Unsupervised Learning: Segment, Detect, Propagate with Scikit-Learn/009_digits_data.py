from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target,
        test_size=test_size,
        random_state=random_state,
        stratify=digits.target
    )
    return X_train, X_test, y_train, y_test