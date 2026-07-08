def average_params(params_list):
    avg = {}
    for key in params_list[0]:
        avg[key] = np.mean([p[key] for p in params_list], axis=0)

    return avg
