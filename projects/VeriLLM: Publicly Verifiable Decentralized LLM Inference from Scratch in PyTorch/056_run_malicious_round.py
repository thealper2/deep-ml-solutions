def run_malicious_round(model_params, prompt_ids, num_steps, verifier_ids, worker_id, committee_size, k, seed, balances, slash_amount, tamper_position, new_token):
    gen = generate_with_state_log(prompt_ids, model_params, num_steps)
    step_states = gen['step_states']
    output_tokens = gen['generated_tokens']

    prev_token = prompt_ids[-1]
    for i, state in enumerate(step_states):
        state['step_index'] = i
        state['input_token'] = prev_token
        prev_token = state['next_token']

    leaves = [commit_decode_step(state) for state in step_states]
    prover_result = {
        'output_tokens': output_tokens,
        'step_states': step_states,
        'leaves': leaves,
    }
    transcript = assemble_public_transcript(prover_result, prompt_ids)

    tampered_transcript = tamper_transcript_flip_token(transcript, tamper_position, new_token)

    committee = sample_verifier_committee(verifier_ids, committee_size, seed)

    votes = collect_verifier_votes(committee, tampered_transcript, model_params, k, seed)

    agg = aggregate_votes_majority(votes)
    verdict = agg['verdict']

    if not verdict:
        new_balances = slash_worker(balances, worker_id, slash_amount)
    else:
        new_balances = balances.copy()

    return {
        'committee': committee,
        'votes': votes,
        'aggregate': agg,
        'accept_count': agg['accept_count'],
        'reject_count': agg['reject_count'],
        'verdict': verdict,
        'balances': new_balances,
        'tampered_transcript': tampered_transcript,
    }
