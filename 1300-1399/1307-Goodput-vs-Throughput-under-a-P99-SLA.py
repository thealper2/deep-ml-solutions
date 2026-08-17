def goodput_metrics(requests, sla_ttft_ms, sla_itl_ms, wall_time_s):
    """
    Throughput vs goodput for a decode trace under a TTFT/ITL SLA.

    Returns a dict with throughput_tps, goodput_tps, n_ok, n_total.
    """
    total_tokens = 0
    good_tokens = 0
    n_ok = 0
    n_total = len(requests)

    for req in requests:
        output_tokens = req['output_tokens']
        total_tokens += output_tokens

        ttft_ok = req['ttft_ms'] <= sla_ttft_ms

        itl_ok = all(itl <= sla_itl_ms for itl in req['itl_ms'])

        if ttft_ok and itl_ok:
            good_tokens += output_tokens
            n_ok += 1

    throughput_tps = total_tokens / wall_time_s
    goodput_tps = good_tokens / wall_time_s

    return {
        'throughput_tps': round(throughput_tps, 4),
        'goodput_tps': round(goodput_tps, 4),
        'n_ok': n_ok,
        'n_total': n_total,
    }
    