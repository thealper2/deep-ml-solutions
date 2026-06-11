def count_client_samples(client_partitions):
    return [features.shape[0] for features, _ in client_partitions]
