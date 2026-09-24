import math


def chunk_latency(chunk_s, rtf, rtt_s):
    return chunk_s + rtf * chunk_s + rtt_s


def largest_chunk_under(budget_s, rtf, rtt_s, step=0.05):
    best = None
    k = 1
    while True:
        chunk = k * step
        if chunk_latency(chunk, rtf, rtt_s) <= budget_s + 1e-9:
            best = chunk
            k += 1

        else:
            break

    if best is None:
        return None

    return round(best, 2)

def concurrent_streams(rtf, gpu_share=1.0):
    return int(math.floor(gpu_share / rtf))


def end_of_speech_latency(hangover_s, chunk_s, rtf, rtt_s):
    return hangover_s + chunk_latency(chunk_s, rtf, rtt_s)


def budget_breakdown(chunk_s, rtf, rtt_s):
    wait = chunk_s
    model = rtf * chunk_s
    network = rtt_s
    total = wait + model + network
    return {
        'wait': round(wait / total, 4),
        'model': round(model / total, 4),
        'network': round(network / total, 4),
    }