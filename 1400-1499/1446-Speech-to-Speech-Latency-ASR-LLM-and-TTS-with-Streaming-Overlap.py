import math


def sequential_latency(asr_s, ttft_s, n_tokens, tps, ttfa_s):
    return asr_s + ttft_s + n_tokens / tps + ttfa_s


def streamed_latency(asr_s, ttft_s, first_sentence_tokens, tps, ttfa_s):
    return asr_s + ttft_s + first_sentence_tokens / tps + ttfa_s


def breakdown(asr_s, ttft_s, tokens, tps, ttfa_s):
    asr = asr_s
    ttft = ttft_s
    generation = tokens / tps
    ttfa = ttfa_s
    total = asr + ttft + generation + ttfa

    components = [
        ('asr', asr),
        ('ttft', ttft),
        ('generation', generation),
        ('ttfa', ttfa),
    ]

    largest = max(components, key=lambda x: x[1])[0]

    return {
        'asr': round(asr, 4),
        'ttft': round(ttft, 4),
        'generation': round(generation, 4),
        'ttfa': round(ttfa, 4),
        'total': round(total, 4),
        'largest': largest,
    }

def first_sentence_budget(target_s, asr_s, ttft_s, tps, ttfa_s):
    fixed = asr_s + ttft_s + ttfa_s
    remaining = target_s - fixed
    if remaining < 0:
        return 0

    n = int(remaining * tps)
    while n > 0 and streamed_latency(asr_s, ttft_s, n, tps, ttfa_s) > target_s + 1e-9:
        n -= 1

    while streamed_latency(asr_s, ttft_s, n + 1, tps, ttfa_s) <= target_s + 1e-9:
        n += 1

    return max(0, n)
