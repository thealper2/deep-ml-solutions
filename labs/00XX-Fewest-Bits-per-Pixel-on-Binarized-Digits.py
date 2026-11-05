import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 8                 # padding around the image
RW, CW = 8, 7         # causal window: RW rows up, +-CW cols
BS, BR, BC = 4, 3, 3  # coarse context blocks: size, rows, half-cols
CFG = dict(K=60, shifts=0, hid=128, hid0=64, frac=0.75, bs=512, bs0=256, lr=2e-3, far_every=4, near=3, ens=1)

_dr, _dc = [], []
for a in range(-RW, 1):
    for b in range(-CW, CW + 1):
        if a < 0 or b < 0:
            _dr.append(a)
            _dc.append(b)
DR, DC = np.array(_dr), np.array(_dc)
NW = len(DR)
NEAR = (np.abs(DR) <= CFG['near']) & (np.abs(DC) <= CFG['near'])
NB = BR * (2 * BC + 1)


def _pad(X):
    n = len(X)
    im = np.zeros((n, 28 + 2 * P, 28 + 2 * P), np.uint8)
    im[:, P:P + 28, P:P + 28] = X.reshape(n, 28, 28)
    return im


def _integral(X28):
    n = len(X28)
    I = np.zeros((n, 29, 29), np.int16)
    I[:, 1:, 1:] = X28.astype(np.int16).cumsum(1).cumsum(2)
    return I


def _coarse(I, idx, r, c):
    """Ink in blocks lying entirely above the window, so they only see past rows."""
    out = np.empty((len(idx), NB), np.float32)
    q = 0
    for a in range(BR):
        r1 = np.clip(r - RW - a * BS, 0, 28)
        r0 = np.clip(r - RW - (a + 1) * BS, 0, 28)
        for b in range(-BC, BC + 1):
            c0 = np.clip(c + b * BS - BS // 2, 0, 28)
            c1 = np.clip(c + b * BS + BS - BS // 2, 0, 28)
            out[:, q] = (I[idx, r1, c1] - I[idx, r0, c1] - I[idx, r1, c0] + I[idx, r0, c0]) / 8.0
            q += 1
    return out


def _feat(im, I, idx, r, c, ml):
    w = im[idx[:, None], P + r[:, None] + DR, P + c[:, None] + DC].astype(np.float32)
    return w, _coarse(I, idx, r, c), ml


class Net(nn.Module):
    def __init__(self, nin, hid, big):
        super().__init__()
        self.emb = nn.Embedding(784, 16)
        self.f1 = nn.Linear(nin + 16 + 1, hid)
        self.f2 = nn.Linear(hid, hid)
        self.f3 = nn.Linear(hid, hid // 2) if big else None
        self.o = nn.Linear(hid // 2 if big else hid, 1)
        self.direct = nn.Embedding(784, 1)  # per-position bias
        self.g = nn.Linear(1, 1)
        nn.init.zeros_(self.direct.weight)
        nn.init.ones_(self.g.weight)
        nn.init.zeros_(self.g.bias)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x, pos, ml):
        h = torch.cat([x, self.emb(pos), ml[:, None] / 4], 1)
        h = F.relu(self.f1(h))
        h = F.relu(self.f2(h)) + h
        if self.f3 is not None:
            h = F.relu(self.f3(h))
        return self.o(h)[:, 0] + self.direct(pos)[:, 0] + self.g(ml[:, None])[:, 0]


def _em(X, K, rng, its, a=0.5):
    N = len(X)
    Xf = X.astype(np.float32)
    r = rng.choice(N, K, replace=N < K)
    mu = np.clip(0.8 * Xf[r] + 0.2 * Xf.mean(0), 0.02, 0.98)
    pi = np.full(K, 1.0 / K)
    for _ in range(its):
        l1, l0 = np.log(mu), np.log1p(-mu)
        lp = Xf @ (l1 - l0).T + l0.sum(1) + np.log(pi)
        lp -= lp.max(1, keepdims=True)
        R = np.exp(lp)
        R /= R.sum(1, keepdims=True)
        nk = R.sum(0)
        mu = np.clip((R.T @ Xf + a * Xf.mean(0)[None] * 4 + a * 0.0) / (nk[:, None] + 4 * a), 1e-3, 1 - 1e-3)
        pi = (nk + 1.0) / (N + K)
    return pi.astype(np.float32), mu.astype(np.float32)


def _mix_table(X, pi, mu, chunk=2000):
    """logit p(x_d=1 | prefix) under the mixture, for every image and pixel."""
    out = np.empty(X.shape, np.float32)
    l1, l0 = np.log(mu), np.log1p(-mu)
    dl = l1 - l0
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk].astype(np.float32)
        lp = np.tile(np.log(pi), (len(x), 1))
        for d in range(784):
            q = lp - lp.max(1, keepdims=True)
            w = np.exp(q)
            p = (w @ mu[:, d]) / w.sum(1)
            p = np.clip(p, 1e-5, 1 - 1e-5)
            out[s:s + chunk, d] = np.log(p / (1 - p))
            lp += x[:, d, None] * dl[:, d] + l0[:, d]
    return out


def _shift(X, a, b):
    im = X.reshape(-1, 28, 28)
    o = np.zeros_like(im)
    o[:, max(a, 0):28 + min(a, 0), max(b, 0):28 + min(b, 0)] = im[:, max(-a, 0):28 + min(-a, 0), max(-b, 0):28 + min(-b, 0)]
    return o.reshape(-1, 784)


def train(X_train):
    t0 = time.process_time()
    cpu = lambda: time.process_time() - t0
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = np.asarray(X_train, dtype=np.uint8)
    N = len(X)
    B = 120.0 if N > 5000 else 60.0
    tb = min(CFG['frac'] * B, B * CFG['frac'] * max(N, 1500) / (10000 if N > 5000 else 5000))
    K = min(CFG['K'], max(2, N // 40))
    # cross-fitted mixtures: training features for a fold come from the other fold's mixture
    perm = rng.permutation(N)
    fa, fb = perm[:N // 2], perm[N // 2:]
    its = 12
    m0 = _em(X[fa], K, rng, its)
    m1 = _em(X[fb], K, rng, its)
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)][:CFG['shifts'] + 1]
    ims, Is, mls, pos_on = [], [], [], np.zeros(784, bool)
    for a, b in shifts:
        Xs = _shift(X, a, b)
        ml = np.empty(X.shape, np.float32)
        ml[fa] = _mix_table(Xs[fa], *m1)
        ml[fb] = _mix_table(Xs[fb], *m0)
        ims.append(_pad(Xs))
        Is.append(_integral(Xs.reshape(-1, 28, 28)))
        mls.append(ml)
        if cpu() > 0.3 * B:
            break
    im = np.concatenate(ims)
    I = np.concatenate(Is)
    ml = np.concatenate(mls)
    del ims, Is, mls
    M = len(im)
    mean = (X.sum(0) + 0.5) / (N + 1.0)
    active = X.sum(0) > 0
    # which (image, pixel) samples have any ink in the local window
    act_pos = np.where(active)[0]
    IM = _integral(im[:, P:P + 28, P:P + 28])
    nact = len(act_pos)
    # list the near-stroke samples: they carry nearly all the bits
    ni, nd = [], []
    for d in act_pos:
        r, c = d // 28, d % 28
        v = im[:, P + r + DR[NEAR], P + c + DC[NEAR]].any(1)
        h = np.where(v)[0].astype(np.int32)
        ni.append(h)
        nd.append(np.full(len(h), d, np.int16))
    ni = np.concatenate(ni)
    nd = np.concatenate(nd)
    n_near = len(ni)
    nets = []
    step = 0
    t_all = cpu()
    E_ = CFG['ens']
    for mem in range(E_):
        net = Net(NW + NB, CFG['hid'], True)
        net0 = Net(NW + NB, CFG['hid0'], False)
        t1 = cpu()
        tb_m = t_all + (tb - t_all) * (mem + 1) / E_
        opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=CFG.get('wd',0.0))
        opt0 = torch.optim.Adam(net0.parameters(), lr=CFG['lr'])
        bs, bs0 = CFG['bs'], CFG['bs0']
        step = 0
        while True:
            now = cpu()
            if now > tb_m:
                break
            frac = (now - t1) / max(tb_m - t1, 1e-9)
            lr = CFG['lr'] * min(1.0, 0.1 + step / 200) * (0.5 * (1 + np.cos(np.pi * frac)) * 0.98 + 0.02)
            for o_ in (opt, opt0):
                for g in o_.param_groups:
                    g['lr'] = lr
            j = rng.integers(n_near, size=bs)
            idx = ni[j]
            d = nd[j].astype(np.int64)
            for which in (0, 1):
                if which == 1:
                    if step % CFG['far_every']:
                        continue
                    idx = rng.integers(M, size=bs0)
                    d = act_pos[rng.integers(nact, size=bs0)]
                r, c = d // 28, d % 28
                w = im[idx[:, None], P + r[:, None] + DR, P + c[:, None] + DC].astype(np.float32)
                if which == 1:
                    keep = w[:, NEAR].sum(1) == 0
                    if not keep.any():
                        continue
                    idx, d, r, c, w = idx[keep], d[keep], r[keep], c[keep], w[keep]
                co = _coarse(IM, idx, r, c)
                y = torch.from_numpy(im[idx, P + r, P + c].astype(np.float32))
                x = torch.from_numpy(np.concatenate([w, co], 1))
                nn_, op = (net, opt) if which == 0 else (net0, opt0)
                o = nn_(x, torch.from_numpy(d), torch.from_numpy(ml[idx, d]))
                loss = F.binary_cross_entropy_with_logits(o, y)
                op.zero_grad()
                loss.backward()
                op.step()
            step += 1
        net.eval()
        net0.eval()
        nets.append((net, net0))
    return dict(nets=nets, m=(m0, m1), mean=mean, active=active, B=B / 2, steps=step,
                state=dict(d=-1, n=-1, lp=None, spent=0.0))


def _mixp(mix, st, X_prefix):
    d = X_prefix.shape[1]
    n = len(X_prefix)
    if d == 0 or st['d'] != d - 1 or st['n'] != n:
        st['lp'] = []
        for pi, mu in mix:
            l1, l0 = np.log(mu), np.log1p(-mu)
            lp = np.tile(np.log(pi), (n, 1))
            if d:
                lp = lp + X_prefix.astype(np.float32) @ (l1[:, :d] - l0[:, :d]).T + l0[:, :d].sum(1)
            st['lp'].append(lp)
    else:
        v = X_prefix[:, -1].astype(np.float32)[:, None]
        for q, (pi, mu) in enumerate(mix):
            a = np.log(mu[:, d - 1])
            b = np.log1p(-mu[:, d - 1])
            st['lp'][q] = st['lp'][q] + v * (a - b) + b
    st['d'], st['n'] = d, n
    ps = 0.0
    for q, (pi, mu) in enumerate(mix):
        lp = st['lp'][q]
        w = np.exp(lp - lp.max(1, keepdims=True))
        ps = ps + (w @ mu[:, d]) / w.sum(1)
    return np.clip(ps / len(mix), 1e-5, 1 - 1e-5).astype(np.float32)


def cond_prob(model, X_prefix):
    t = time.process_time()
    st = model['state']
    d = X_prefix.shape[1]
    n = len(X_prefix)
    pm = _mixp(model['m'], st, X_prefix)
    if d == 0:
        st['spent'] = 0.0
    if not model['active'][d]:
        st['spent'] += time.process_time() - t
        return np.full(n, model['mean'][d])
    if st['spent'] > 0.8 * model['B']:   # emergency fallback, should not trigger
        st['spent'] += time.process_time() - t
        return pm.astype(np.float64)
    X = np.zeros((n, 784), np.uint8)
    X[:, :d] = X_prefix
    r, c = d // 28, d % 28
    if st.get('Ir') != r or len(st['I']) != n:  # coarse blocks only change with the row
        st['I'] = _integral(X.reshape(n, 28, 28))
        st['Ir'] = r
    im = _pad(X)
    idx = np.arange(n)
    rr, cc = np.full(n, r), np.full(n, c)
    w, co, _ = _feat(im, st['I'], idx, rr, cc, None)
    ml = np.log(pm / (1 - pm))
    near = torch.from_numpy(w[:, NEAR].sum(1) > 0)
    far = ~near
    x = torch.from_numpy(np.concatenate([w, co], 1))
    tm = torch.from_numpy(ml)
    tp = torch.full((n,), d, dtype=torch.long)
    o = torch.zeros(n)
    with torch.no_grad():
        for net, net0 in model['nets']:
            if near.any():
                o[near] += net(x[near], tp[near], tm[near])
            if far.any():
                o[far] += net0(x[far], tp[far], tm[far])
    o /= len(model['nets'])
    p = torch.sigmoid(o).numpy().astype(np.float64)
    st['spent'] += time.process_time() - t
    return np.clip(p, 1e-5, 1 - 1e-5)