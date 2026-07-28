def compare_with_normal_equation(model):
    return weights_l2_distance(model['weights'], model['normal_weights'])
