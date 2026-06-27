def merkle_inclusion_proof(tree, leaf_index):
    proof = []
    idx = leaf_index

    for level in range(len(tree) - 1):
        nodes = tree[level]
        if idx % 2 == 0:
            if idx + 1 < len(nodes):
                sibling = nodes[idx + 1]
            else:
                sibling = nodes[idx]
            
            is_right = True

        else:
            sibling = nodes[idx - 1]
            is_right = False

        proof.append({'sibling': sibling, 'is_right': is_right})
        idx = idx // 2

    return proof
