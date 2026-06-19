def lenet_forward(x, params):
    block1_out, block1_cache = forward_conv_block(
        x,
        params['conv1']['W'],
        params['conv1']['b'],
        pool_size=2,
        stride=1,
        pad=0,
    )
    block2_out, block2_cache = forward_conv_block(
        block1_out,
        params['conv2']['W'],
        params['conv2']['b'],
        pool_size=2,
        stride=1,
        pad=0,
    )
    logits, classifier_cache = forward_classifier_block(
        block2_out,
        params['fc1'],
        params['fc2'],
    )
    caches = {
        'block1': block1_cache,
        'block2': block2_cache,
        'classifier': classifier_cache,
    }
    return logits, caches
