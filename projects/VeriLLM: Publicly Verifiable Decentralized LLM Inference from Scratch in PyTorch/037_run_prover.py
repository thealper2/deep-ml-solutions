def run_prover(model_params, prompt_ids, num_steps):
    if num_steps <= 0:
        return {'output_tokens': [], 'step_states': [], 'leaves': []}

    gen = generate_with_state_log(prompt_ids, model_params, num_steps)
    output_tokens = gen['generated_tokens']
    step_states = gen['step_states']

    leaves = [commit_decode_step(state) for state in step_states]

    return {
        'output_tokens': output_tokens,
        'step_states': step_states,
        'leaves': leaves,
    }
