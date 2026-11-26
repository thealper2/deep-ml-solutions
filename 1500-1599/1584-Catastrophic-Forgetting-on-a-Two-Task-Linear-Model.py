import numpy as np


def ridge_fit(X, y, lam):
    """Closed-form ridge regression weights (no intercept)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _mse(X, y, w):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    r = X @ w - y
    return float(np.mean(r * r))


def sequential_forgetting(XA, yA, XB, yB, lam=0.0, decimals=4):
    """
    Fit on A, fit on B, fit on A+B jointly. Report task-A and task-B MSEs and the forgetting
    mse_A_after_B - mse_A_after_A. All values rounded to `decimals`, as Python floats.
    """
    wA = ridge_fit(XA, yA, lam)
    wB = ridge_fit(XB, yB, lam)

    XJ = np.vstack([np.asarray(XA, dtype=float), np.asarray(XB, dtype=float)])
    yJ = np.concatenate([np.asarray(yA, dtype=float), np.asarray(yB, dtype=float)])
    wJ = ridge_fit(XJ, yJ, lam)

    mse_A_after_A = _mse(XA, yA, wA)
    mse_A_after_B = _mse(XA, yA, wB)
    mse_B_after_B = _mse(XB, yB, wB)
    forgetting = mse_A_after_B - mse_A_after_A
    mse_A_joint = _mse(XA, yA, wJ)
    mse_B_joint = _mse(XB, yB, wJ)
    joint_total = mse_A_joint + mse_B_joint

    return {
        "mse_A_after_A": round(mse_A_after_A, decimals),
        "mse_A_after_B": round(mse_A_after_B, decimals),
        "mse_B_after_B": round(mse_B_after_B, decimals),
        "mse_A_joint": round(mse_A_joint, decimals),
        "mse_B_joint": round(mse_B_joint, decimals),
        "forgetting": round(forgetting, decimals),
        "joint_total": round(joint_total, decimals),
    }