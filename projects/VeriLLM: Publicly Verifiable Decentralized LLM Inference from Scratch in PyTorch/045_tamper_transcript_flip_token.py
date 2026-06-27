def tamper_transcript_flip_token(transcript, position, new_token):
    new_transcript = dict(transcript)
    output_tokens = transcript['output_tokens']
    new_transcript['output_tokens'] = (
        output_tokens.copy() if hasattr(output_tokens, 'copy') else output_tokens[:]
    )
    new_transcript['output_tokens'][position] = new_token
    return new_transcript
