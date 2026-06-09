import jax

def one_hot_encode_labels(labels, num_classes):
    one_hot_matrix = jax.nn.one_hot(labels, num_classes)
    return one_hot_matrix
