def normal_equation(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)
