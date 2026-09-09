from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test