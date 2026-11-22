def launch_dims(shape, block=(32, 4, 1)):
    extents = list(shape)
    e = [1, 1, 1]
    n = len(extents)
    for i in range(min(n, 3)):
        e[i] = extents[n - 1 - i]

    b = []
    for i in range(3):
        b.append(min(block[i], e[i]))

    g = []
    for i in range(3):
        g.append(-(-e[i] // b[i]))

    return (tuple(g), tuple(b))


def threads_launched(grid, block):
    return grid[0] * block[0] * grid[1] * block[1] * grid[2] * block[2]


def utilization(shape, block=(32, 4, 1)):
    grid, b = launch_dims(shape, block)
    launched = threads_launched(grid, b)
    useful = 1
    for s in shape:
        useful *= s
    return useful / launched


def global_index(block_id, thread_id, block):
    return tuple(block_id[i] * block[i] + thread_id[i] for i in range(3))


def in_bounds(gidx, shape):
    n = len(shape)
    for i in range(3):
        axis = n - 1 - i
        if axis >= 0:
            if gidx[i] >= shape[axis]:
                return False
        else:
            if gidx[i] != 0:
                return False
    return True


def best_block(shape, candidates):
    best = None
    best_u = None
    for c in candidates:
        u = utilization(shape, c)
        if best_u is None or u > best_u:
            best_u = u
            best = c
    return best