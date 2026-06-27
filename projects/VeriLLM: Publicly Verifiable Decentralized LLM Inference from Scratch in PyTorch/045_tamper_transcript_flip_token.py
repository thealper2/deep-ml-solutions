def tamper_transcript_flip_token(transcript, position, new_token):
    new_transcript = {
        'prompt_ids': transcript['prompt_ids'].copy() if hasattr(transcript['prompt_ids'], 'copy') else transcript['prompt_ids'][:],
        'output_tokens': transcript['output_tokens'].copy() if hasattr(transcript['output_tokens'], 'copy') else transcript['output_tokens'][:],
        'leaves': transcript['leaves'],
        'tree': transcript['tree'],
        'root': transcript['root'],
        'step_states': transcript['step_states'].copy() if hasattr(transcript['step_states'], 'copy') else transcript['step_states'][:]
    }
    new_transcript['output_tokens'][position] = new_token
    return new_transcript
