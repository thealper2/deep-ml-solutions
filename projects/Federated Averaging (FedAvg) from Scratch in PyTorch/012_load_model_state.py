def load_model_state(model, state_dict):
    model.load_state_dict(state_dict)
    return model
