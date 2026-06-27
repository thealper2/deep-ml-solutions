def assemble_public_transcript(prover_result, prompt_ids):
    leaves = prover_result['leaves']
    tree = build_merkle_tree(leaves)
    root = merkle_root(tree)
    
    return {
        'prompt_ids': prompt_ids.copy() if hasattr(prompt_ids, 'copy') else prompt_ids[:],
        'output_tokens': prover_result['output_tokens'].copy() if hasattr(prover_result['output_tokens'], 'copy') else prover_result['output_tokens'][:],
        'leaves': prover_result['leaves'].copy() if hasattr(prover_result['leaves'], 'copy') else prover_result['leaves'][:],
        'tree': tree,
        'root': root,
        'step_states': prover_result['step_states'].copy() if hasattr(prover_result['step_states'], 'copy') else prover_result['step_states'][:]
    }
