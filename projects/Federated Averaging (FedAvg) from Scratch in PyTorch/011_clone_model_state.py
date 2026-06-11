def clone_model_state(model):
    state_dict = model.state_dict()
    cloned_dict = {}
    for name, tensor in state_dict.items():
        cloned_dict[name]= tensor.detach().clone()

    return cloned_dict
