import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    params = []
    seen = set()

    for layer in encoder_layer_params:
        for tensor in layer.values():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                params.append(tensor)

    for layer in decoder_layer_params:
        for tensor in layer.values():
            if id(tensor) not in seen:
                seen.add(id(tensor))
                params.append(tensor)

    for tensor in embedding_params.values():
        if id(tensor) not in seen:
            seen.add(id(tensor))
            params.append(tensor)

    return params
