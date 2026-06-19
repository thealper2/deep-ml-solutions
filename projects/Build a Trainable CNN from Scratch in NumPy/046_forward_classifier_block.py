def forward_classifier_block(x, fc1, fc2):
    flat_out, flatten_cache = flatten_forward(x)
    fc1_out, fc1_cache = linear_forward(flat_out, fc1['W'], fc1['b'])
    relu_out, relu_cache = relu_forward(fc1_out)
    logits, fc2_cache = linear_forward(relu_out, fc2['W'], fc2['b'])
    cache = {
        'flatten_cache': flatten_cache,
        'fc1_cache': fc1_cache,
        'relu_cache': relu_cache,
        'fc2_cache': fc2_cache,
    }
    return logits, cache
