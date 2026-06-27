def collect_verifier_votes(committee, transcript, model_params, k, base_seed):
    votes = []

    for verifier_id in committee:
        seed = hash(verifier_id) ^ base_seed
        if seed < 0:
            seed = -seed

        seed = seed % (2 ** 31)

        result = run_spot_check_verification(transcript, model_params, seed, k)

        votes.append({
            'verifier_id': verifier_id,
            'vote': result['accept'],
            'result': result,
        })

    return votes
