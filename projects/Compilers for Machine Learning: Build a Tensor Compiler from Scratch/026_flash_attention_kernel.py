import math


def flash_attention_kernel(N, d, name="flash"):
    out = param("out", dtypes.float32, 0)
    q = param("q", dtypes.float32, 1)
    kk = param("k", dtypes.float32, 2)
    v = param("v", dtypes.float32, 3)

    i = UOp.range(N, 0)
    j = UOp.range(N, 1)
    c = UOp.range(d, 2)

    # nested reduce over d: s = sum_c q[i,c] * k[j,c]
    s_acc = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 3)
    q_load = load(q, i * d + c)
    k_load = load(kk, j * d + c)
    s_update = s_acc + q_load * k_load
    inner = Reduce([c], [[s_acc, UOp.const(dtypes.float32, 0.0), s_update]], [])

    s = s_acc * (1.0 / (d ** 0.5))

    # running max / sum / columns
    m_acc = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 4)
    m_new = m_acc.maximum(s)
    alpha = ((m_acc - m_new) * (1.0 / math.log(2))).exp2()
    p = ((s - m_new) * (1.0 / math.log(2))).exp2()
    m_accs = [m_acc, UOp.const(dtypes.float32, -INF), m_new]

    l_acc = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 5)
    l_accs = [l_acc, UOp.const(dtypes.float32, 0.0), l_acc * alpha + p]

    accs = [m_accs, l_accs]
    col_accs = []
    for col in range(d):
        a = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 6 + col)
        v_load = load(v, j * d + col)
        accs.append([a, UOp.const(dtypes.float32, 0.0), a * alpha + p * v_load])
        col_accs.append(a)

    outer = Reduce([j], accs, [inner])

    stores = [(i * d + col, col_accs[col] * l_acc.recip()) for col in range(d)]
    return Kernel(name, [out, q, kk, v], [i], [outer], stores)