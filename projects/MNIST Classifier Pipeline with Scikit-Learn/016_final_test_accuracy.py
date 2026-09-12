from sklearn.metrics import accuracy_score

def final_test_accuracy(model, X_test, y_test):
    return float(accuracy_score(y_test, model.predict(X_test)))