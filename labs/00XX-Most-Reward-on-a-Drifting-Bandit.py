import numpy as np


class Agent:
    """Kalman tracking of every arm under a bank of random-walk drift speeds.

    Each arm of each problem has a Gaussian belief (mean m, variance P). Reward noise has
    variance 1, and between pulls an arm's variance grows by q per step. Instead of touching
    every arm on every step, an arm is brought forward lazily from the step it was last pulled,
    so a step costs O(n k) and a pull updates only that arm, under every drift speed q at once.
    Each problem follows the q with the best discounted predictive log-likelihood of its own
    rewards, which lets one agent handle slow walks, fast mean-reverting values and jumps.
    Actions maximise m + sqrt(P) * (C z + B): a little Thompson noise plus a fixed optimism bonus.
    """
    QS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    C, B = 0.3, 0.6
    G = 0.9999  # likelihood memory of about 10,000 steps

    def __init__(self, n_problems, k, n_steps):
        self.n, self.k = n_problems, k
        self.q = np.array(self.QS)
        self.qc = self.q[:, None]
        M = len(self.QS)
        self.m = np.zeros((M, n_problems, k))
        self.P = np.ones((M, n_problems, k))
        self.last = np.zeros((n_problems, k))
        self.L = np.zeros((M, n_problems))
        self.t = 0
        self.rows = np.arange(n_problems)
        self.rng = np.random.default_rng(0)
        self.sel = np.full(n_problems, 3, dtype=np.int64)

    def act(self):
        s, rr = self.sel, self.rows
        P = self.P[s, rr] + self.q[s][:, None] * (self.t - self.last)
        z = self.C * self.rng.standard_normal((self.n, self.k)) + self.B
        return (self.m[s, rr] + np.sqrt(P) * z).argmax(1).astype(np.int64)

    def update(self, actions, rewards):
        rr = self.rows
        a = np.asarray(actions, dtype=np.int64)
        r = np.asarray(rewards, dtype=float)
        P = self.P[:, rr, a] + self.qc * (self.t - self.last[rr, a])
        m = self.m[:, rr, a]
        S = P + 1.0
        e = r - m
        self.L *= self.G
        self.L -= 0.5 * (np.log(S) + e * e / S)
        K = P / S
        self.m[:, rr, a] = m + K * e
        self.P[:, rr, a] = P - K * P
        self.t += 1
        self.last[rr, a] = self.t
        self.sel = self.L.argmax(0)
