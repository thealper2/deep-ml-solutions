from sklearn.model_selection import train_test_split

def train_test(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test