def build_merkle_tree(leaves):
    tree = [leaves]
    current_level = leaves
    while len(current_level) > 1:
        current_level = build_merkle_level(current_level)
        tree.append(current_level)

    return tree
