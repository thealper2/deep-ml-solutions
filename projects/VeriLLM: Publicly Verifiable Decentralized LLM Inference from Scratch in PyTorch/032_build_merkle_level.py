def build_merkle_level(nodes):
    parents = []
    n = len(nodes)
    for i in range(0, n, 2):
        left = nodes[i]
        if i + 1 < n:
            right = nodes[i + 1]
        else:
            right = nodes[i]

        parents.append(hash_pair(left, right))

    return parents
