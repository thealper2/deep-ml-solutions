def show_tampered_transcript_rejected(transcript, model_params, position, new_token, seed, k):
    tampered = tamper_transcript_flip_token(transcript, position, new_token)
    result = run_spot_check_verification(tampered, model_params, seed, k)
    
    return {
        'tampered_transcript': tampered,
        'result': result,
        'rejected': not result['accept']
    }
