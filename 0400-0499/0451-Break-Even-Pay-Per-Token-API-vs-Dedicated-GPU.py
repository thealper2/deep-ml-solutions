import math

def break_even_analysis(cost_per_token, gpu_cost_per_hour, max_tokens_per_second, monthly_tokens, hours_per_month=730):
    single_gpu_cost = gpu_cost_per_hour * hours_per_month
    api_cost = round(monthly_tokens * cost_per_token, 2)
    break_even_tokens = round(single_gpu_cost / cost_per_token)

    gpu_can_handle = max_tokens_per_second * 3600 * hours_per_month
    num_gpus = math.ceil(monthly_tokens / gpu_can_handle) if gpu_can_handle > 0 else 1
    if num_gpus < 1:
        num_gpus = 1
    gpu_cost = round(single_gpu_cost * num_gpus, 2)

    if gpu_cost < api_cost:
        recommendation = 'gpu'
    elif api_cost < gpu_cost:
        recommendation = 'api'
    else:
        recommendation = 'equal'

    return {
        'break_even_tokens': break_even_tokens,
        'api_cost': api_cost,
        'gpu_cost': gpu_cost,
        'num_gpus_needed': num_gpus,
        'recommendation': recommendation,
    }