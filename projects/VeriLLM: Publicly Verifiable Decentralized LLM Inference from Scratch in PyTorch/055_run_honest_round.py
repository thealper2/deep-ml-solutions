def run_honest_round(model_params, prompt_ids, num_steps, verifier_ids, worker_id, committee_size, k, seed, balances, reward_worker, reward_verifier):
    prover_result = run_prover(model_params, prompt_ids, num_steps)

    transcript = assemble_public_transcript(prover_result, prompt_ids)

    committee = sample_verifier_committee(verifier_ids, committee_size, seed)

    votes = collect_verifier_votes(committee, transcript, model_params, k, seed)

    agg = aggregate_votes_majority(votes)
    verdict = agg['verdict']

    new_balances = reward_honest_participants(
        balances, worker_id, votes, verdict, reward_worker, reward_verifier
    )

    return {
        'transcript': transcript,
        'votes': votes,
        'verdict': verdict,
        'balances': new_balances,
    }
