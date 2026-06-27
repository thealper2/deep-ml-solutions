def run_spot_check_verification(transcript, model_params, seed, k):
    """Run end-to-end spot-check verification of a prover transcript.

    Returns a dict with keys 'accept', 'audited_positions', 'per_audit'.
    """
    num_steps = len(transcript['output_tokens'])
    audited_positions = sample_audit_positions(seed, num_steps, k)
    
    per_audit = []
    all_passed = True
    
    for pos in audited_positions:
        step_state = transcript['step_states'][pos]
        
        if pos == 0:
            prior_kv_cache = step_state['kv_caches']
            prior_token = step_state['input_token']
        else:
            prev_state = transcript['step_states'][pos - 1]
            prior_kv_cache = prev_state['kv_caches']
            prior_token = prev_state['next_token']
        
        reexec_result = reexecute_audited_step(model_params, prior_kv_cache, prior_token)
        
        recomputed_leaf = recompute_step_commitment(reexec_result, prior_kv_cache)
        
        leaf_index = pos
        proof = merkle_inclusion_proof(transcript['tree'], leaf_index)
        commitment_ok = check_commitment_against_proof(recomputed_leaf, leaf_index, proof, transcript['root'])
        
        claimed_token = transcript['output_tokens'][pos]
        token_ok = check_token_matches_claim(reexec_result['token'], claimed_token)
        
        per_audit.append({
            'commitment_ok': commitment_ok,
            'token_ok': token_ok
        })
        
        if not commitment_ok or not token_ok:
            all_passed = False
    
    return {
        'accept': all_passed,
        'audited_positions': audited_positions,
        'per_audit': per_audit
    }
