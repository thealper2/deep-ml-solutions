def verify_merkle_inclusion_proof(leaf, leaf_index, proof, root):
    current = leaf
    idx = leaf_index

    for entry in proof:
        sibling = entry['sibling']

        if entry.get('is_right', entry.get('side')) == 'right' or entry.get('is_right') == True:
            combined = current + sibling
        else:
            combined = sibling + current

        current = hashlib.sha256(combined).digest()
        idx = idx // 2

    return current == root
