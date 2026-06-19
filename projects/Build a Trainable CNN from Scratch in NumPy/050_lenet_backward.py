def lenet_backward(dlogits, caches):
    classifier_grads = backward_classifier_block(dlogits, caches['classifier'])
    d_block2, dW2, db2 = backward_conv_block(classifier_grads['dx'], caches['block2'])
    d_block1, dW1, db1 = backward_conv_block(d_block2, caches['block1'])

    return {
        'conv1': {'dW': dW1, 'db': db1},
        'conv2': {'dW': dW2, 'db': db2},
        'fc1': {'dW': classifier_grads['fc1']['dW'], 'db': classifier_grads['fc1']['db']},
        'fc2': {'dW': classifier_grads['fc2']['dW'], 'db': classifier_grads['fc2']['db']},
    }
