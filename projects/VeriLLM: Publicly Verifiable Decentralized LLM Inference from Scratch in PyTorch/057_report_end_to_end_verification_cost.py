def report_end_to_end_verification_cost(num_steps, committee_size, k):
    per_verifier_fraction = verifier_cost_fraction(num_steps, k)
    committee_fraction = per_verifier_fraction * committee_size
    full_reexec_fraction = 1.0
    return {
        'per_verifier_fraction': per_verifier_fraction,
        'committee_fraction': committee_fraction,
        'full_reexec_fraction': full_reexec_fraction
    }
