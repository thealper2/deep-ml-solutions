import numpy as np

def build_target_network_copy(online_params):
    """Return a deep copy of the online MLP parameter dict."""
    return {key: value.copy() for key, value in online_params.items()}
