def cost_per_million(gpu_hourly, tokens_per_s, utilization=1.0):
    return round(gpu_hourly / (tokens_per_s * 3600) * 1e6 / utilization, 4)


def blended_price(in_tokens, out_tokens, in_price_m, out_price_m):
    total = in_tokens + out_tokens
    if total == 0:
        return 0.0

    cost = in_tokens * in_price_m + out_tokens * out_price_m
    return round(cost / total * 1e6 / 1e6, 4)

def monthly_dedicated(gpu_hourly, replicas, hours=730):
    return gpu_hourly * replicas * hours


def monthly_api(monthly_tokens_m, price_m):
    return monthly_tokens_m * price_m


def cheapest_option(options, monthly_tokens_m):
    best_name = None
    best_cost = None
    required_tokens = monthly_tokens_m * 1e6

    for opt in options:
        if opt['type'] == 'api':
            cost = monthly_api(monthly_tokens_m, opt['price_m'])
        elif opt['type'] == 'dedicated':
            capacity = opt['replicas'] * opt['tokens_per_s'] * 3600 * 730
            if capacity < required_tokens:
                continue

            cost = monthly_dedicated(opt['gpu_hourly'], opt['replicas'])
        else:
            continue

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_name = opt['name']

    return best_name