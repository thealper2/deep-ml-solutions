def verify_merkle_inclusion_proof(leaf, leaf_index, proof, root):
    current = leaf
    idx = leaf_index

    for entry in proof:
        sibling = entry['sibling']
        side = entry['side']

        if side == 'right':
            combined = current + sibling
        else:
            combined = sibling + current

        current = hashlib.sha256(combined).digest()
        idx = idx // 2

    return current == root
