def check_commitment_against_proof(recomputed_leaf, leaf_index, proof, root):
    return verify_merkle_inclusion_proof(recomputed_leaf, leaf_index, proof, root)
