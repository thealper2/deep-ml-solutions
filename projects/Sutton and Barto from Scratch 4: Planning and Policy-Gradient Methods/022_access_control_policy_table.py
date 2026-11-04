import numpy as np


def access_control_policy_table(q_table):
    q = np.asarray(q_table)
    n_free, n_prio = q.shape[0], q.shape[1]
    table = np.zeros((n_prio, n_free), dtype=int)
    for prio in range(n_prio):
        for free in range(1, n_free):
            if q[free, prio, 1] > q[free, prio, 0]:
                table[prio, free] = 1
    return table