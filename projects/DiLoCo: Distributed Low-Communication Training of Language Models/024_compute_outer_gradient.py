def compute_outer_gradient(global_params, worker_params_list):
    avg_worker = average_params(worker_params_list)
    return subtract_params(global_params, avg_worker)
