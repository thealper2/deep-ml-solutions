import time
import numpy as np


def _logdens(Xf, pi, mu, W, psi):
    N, D = Xf.shape
    out = np.empty((N, len(pi)), np.float32)
    for k in range(len(pi)):
        R = Xf - mu[k].astype(np.float32)
        ip = 1.0 / psi[k]
        q = (R * R * ip.astype(np.float32)).sum(1, dtype=np.float64)
        ld = np.log(psi[k]).sum()
        L = W[k].shape[1]
        if L:
            A = W[k] * ip[:, None]
            c = np.linalg.cholesky(np.eye(L) + W[k].T @ A)
            ld += 2 * np.log(np.diag(c)).sum()
            z = R @ np.linalg.solve(c, A.T).T.astype(np.float32)
            q = q - (z * z).sum(1, dtype=np.float64)
        out[:, k] = np.log(pi[k] + 1e-300) - 0.5 * (D * np.log(2 * np.pi) + ld + q)
    return out


def _resp(lp):
    t = lp.max(1, keepdims=True)
    e = np.exp(np.maximum(lp - t, -30))
    s = e.sum(1, keepdims=True)
    return e / s, t[:, 0].astype(np.float64) + np.log(s[:, 0])


def _fa(S, psi, L, floor, its=3):
    """Alternate the closed-form W given psi with psi given W, on covariance S."""
    W = np.zeros((len(psi), L))
    lam = None
    for _ in range(its):
        s = np.sqrt(psi)
        lam, U = np.linalg.eigh(S / np.outer(s, s))
        lam = lam[::-1]
        if L:
            W = s[:, None] * U[:, ::-1][:, :L] * np.sqrt(np.maximum(lam[:L] - 1, 0))
        psi = np.maximum(np.diag(S) - (W * W).sum(1), floor)
    return W, psi, lam


def _stats(X, w, mu0, var0, a):
    idx = np.where(w > 1e-3)[0]
    n = float(w[idx].sum(dtype=np.float64))
    if len(idx) < 3 or n < 2.0:
        return max(n, 1e-6), mu0.copy(), np.diag(var0)
    ww = w[idx].astype(np.float64)
    x = X[idx]
    m = ww @ x / n
    R = x - m
    S = (R * ww[:, None]).T @ R / n
    return n, m, (n * S + 0.1 * a * np.diag(var0)) / (n + a)


def _alloc(Ss, ns, psis, Ls, cap, floor, pen, D):
    """Hand out factors greedily by their closed-form likelihood gain, minus a penalty."""
    G = []
    for q in range(len(Ss)):
        lam = _fa(Ss[q], psis[q], Ls[q], floor, its=1)[2]
        l = np.maximum(lam[:min(150, int(ns[q] / 4) + 1)], 1.0)
        G += [(g, q) for g in ns[q] * 0.5 * (l - 1 - np.log(l)) - pen * D]
    G.sort(key=lambda t: -t[0])
    out = [0] * len(Ss)
    for g, q in G[:max(cap, 0)]:
        if g > 0:
            out[q] += 1
    return out


def _kmeans(x, K, rng, its=15):
    c = x[rng.choice(len(x), K, replace=False)].copy()
    lab = np.zeros(len(x), int)
    for _ in range(its):
        d = (x * x).sum(1, keepdims=True) - 2 * x @ c.T + (c * c).sum(1)
        lab = d.argmin(1)
        far = np.argsort(-d.min(1))
        f = 0
        for k in range(K):
            if (lab == k).any():
                c[k] = x[lab == k].mean(0)
            else:
                c[k] = x[far[f]]
                f += 1
    return c


def fit(X_train):
    t0 = time.process_time()
    X = np.asarray(X_train, dtype=np.float64)
    N, D = X.shape
    cpu = lambda: time.process_time() - t0
    # Budgets are 60 s on Run and 180 s on Submit. Stay well inside them, since
    # threaded BLAS can burn CPU time faster than the single-thread tests here.
    B = 180.0 if N > 10000 else 60.0
    t_init = 0.25 * B
    t_search = min(0.55 * B, 110.0 * N / 20000)
    t_end = min(0.68 * B, t_search + 15.0)
    budget, pen, a = 48000, 2.0, 20.0
    floor = 3.3e-4  # just above var of the uniform noise, (1/16)^2 / 12
    rng = np.random.default_rng(0)
    K = max(2, min(15, N // 800))
    K = min(K, max(1, N // 2))
    cap = (budget - K - 2 * K * D) // D
    Xf = X.astype(np.float32)
    mu0, var0 = X.mean(0), np.maximum(X.var(0), floor)

    sub = Xf if N <= 8000 else Xf[rng.choice(N, 8000, replace=False)]
    c = _kmeans(sub, K, rng).astype(np.float64)
    lab = ((Xf * Xf).sum(1, keepdims=True)
           - 2 * Xf @ c.T.astype(np.float32) + (c * c).sum(1).astype(np.float32)).argmin(1)
    pi = np.maximum(np.bincount(lab, minlength=K), 1) / N
    pi = pi / pi.sum()
    mu = np.empty((K, D))
    psi = np.empty((K, D))
    for k in range(K):
        x = X[lab == k]
        mu[k] = x.mean(0) if len(x) > 1 else mu0
        psi[k] = np.maximum(x.var(0), floor) if len(x) > 1 else var0
    W = [np.zeros((D, 0)) for _ in range(K)]

    def em_step():
        nonlocal pi
        r = _resp(_logdens(Xf, pi, mu, W, psi))[0]
        Ss, ns = [], []
        for k in range(K):
            n, mu[k], S = _stats(X, r[:, k], mu0, var0, a)
            Ss.append(S)
            ns.append(n)
        pi = np.array(ns) / sum(ns)
        Ls = _alloc(Ss, ns, list(psi), [w.shape[1] for w in W], cap, floor, pen, D)
        for k in range(K):
            W[k], psi[k], _ = _fa(Ss[k], psi[k], Ls[k], floor)

    for _ in range(15):
        em_step()
        if cpu() > t_init:
            break

    # Split-merge search: merge two similar components, split a third, keep if better.
    lp = _logdens(Xf, pi, mu, W, psi)
    r, ll = _resp(lp)
    best = ll.mean()
    acc = 0
    while K >= 3 and cpu() < t_search:
        nk = r.sum(0).astype(np.float64)
        v = np.stack([(W[k] * W[k]).sum(1) + psi[k] for k in range(K)])
        lv = nk * np.log(v).sum(1)
        M = np.full((K, K), np.inf)
        for i in range(K):
            for j in range(i + 1, K):
                t = nk[i] + nk[j] + 1e-9
                f = nk[i] / t
                vm = f * v[i] + (1 - f) * v[j] + f * (1 - f) * (mu[i] - mu[j]) ** 2
                M[i, j] = t * np.log(vm).sum() - lv[i] - lv[j]
        o = np.argsort(M, axis=None)[:min(25, K * (K - 1) // 2)]
        i, j = np.unravel_index(o[rng.integers(len(o))], (K, K))
        cand = [k for k in range(K) if k != i and k != j]
        s = cand[rng.choice(len(cand), p=(nk[cand] + 1e-9) / (nk[cand] + 1e-9).sum())]
        ix = np.where(r[:, s] > 0.1)[0]
        if len(ix) < 40:
            continue
        if rng.random() < 0.5:  # split off the tail of the component
            y = lp[ix, s]
            lb = y < np.quantile(y, rng.uniform(0.1, 0.4))
        else:  # 2-means seeded along a random leading direction
            x = Xf[ix]
            d = np.linalg.eigh(np.cov(x.T))[1][:, -1 - rng.integers(4)].astype(np.float32)
            pr = (x - x.mean(0)) @ d
            lb = pr > np.median(pr)
            for _ in range(6):
                if lb.sum() < 5 or (~lb).sum() < 5:
                    break
                lb = ((x - x[lb].mean(0)) ** 2).sum(1) < ((x - x[~lb].mean(0)) ** 2).sum(1)
        if lb.sum() < 15 or (~lb).sum() < 15:
            continue
        C = [i, j, s]
        wa, wb = np.zeros(N, np.float32), np.zeros(N, np.float32)
        wa[ix[~lb]] = r[ix[~lb], s]
        wb[ix[lb]] = r[ix[lb], s]
        st = [_stats(X, w, mu0, var0, a) for w in (r[:, i] + r[:, j], wa, wb)]
        Lold = [W[q].shape[1] for q in C]
        sav = (pi.copy(), mu[C].copy(), [W[q] for q in C], psi[C].copy(), lp[:, C].copy())
        tot = pi[C].sum()
        pm = (nk[i] * psi[i] + nk[j] * psi[j]) / (nk[i] + nk[j] + 1e-9)
        pl = [np.maximum(pm, floor), psi[s].copy(), psi[s].copy()]
        Lc = None
        for e in range(3):
            ns = [t[0] for t in st]
            if Lc is None or e == 2:
                Lc = _alloc([t[2] for t in st], ns, pl, Lold, sum(Lold), floor, pen, D)
            for q, k in enumerate(C):
                pi[k] = tot * ns[q] / sum(ns)
                mu[k] = st[q][1]
                W[k], psi[k], _ = _fa(st[q][2], pl[q], Lc[q], floor, its=2)
                pl[q] = psi[k]
            lp[:, C] = _logdens(Xf, pi[C], mu[C], [W[k] for k in C], psi[C])
            rr, llt = _resp(lp)
            st = [_stats(X, rr[:, k], mu0, var0, a) for k in C]
        if llt.mean() > best:
            r, ll, best = rr, llt, llt.mean()
            acc += 1
            if acc % 6 == 0:
                em_step()
                em_step()
                lp = _logdens(Xf, pi, mu, W, psi)
                r, ll = _resp(lp)
                best = ll.mean()
        else:
            pi[:] = sav[0]
            mu[C] = sav[1]
            psi[C] = sav[3]
            lp[:, C] = sav[4]
            for q, k in enumerate(C):
                W[k] = sav[2][q]

    for _ in range(8):
        if cpu() > t_end:
            break
        em_step()

    pi = np.maximum(pi, 1e-12)
    return {"pi": pi / pi.sum(), "mu": mu.copy(), "W": [w.copy() for w in W],
            "psi": np.maximum(np.nan_to_num(psi, nan=floor), floor)}