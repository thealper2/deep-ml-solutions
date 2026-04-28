import numpy as np

def validate_space(space: dict, samples: list) -> list:
    """
    Validate whether each sample conforms to the given space definition.
    
    Args:
        space: A dictionary defining the space with a 'type' key and type-specific parameters.
        samples: A list of samples to validate against the space.
    
    Returns:
        A list of booleans indicating whether each sample is valid for the space.
    """
    results = []
    
    for sample in samples:
        valid = False
        
        if space["type"] == "Discrete":
            n = space["n"]
            valid = isinstance(sample, int) and 0 <= sample < n
        
        elif space["type"] == "Box":
            low = np.array(space["low"])
            high = np.array(space["high"])
            expected_shape = tuple(space["shape"])
            sample_arr = np.array(sample)
            if sample_arr.shape == expected_shape:
                valid = np.all((sample_arr >= low) & (sample_arr <= high))
        
        elif space["type"] == "MultiBinary":
            n = space["n"]
            sample_arr = np.array(sample)
            if sample_arr.shape == (n,):
                valid = np.all((sample_arr == 0) | (sample_arr == 1))
        
        elif space["type"] == "MultiDiscrete":
            nvec = np.array(space["nvec"])
            sample_arr = np.array(sample)
            expected_shape = tuple(nvec.shape)
            if sample_arr.shape == expected_shape:
                valid = np.all((sample_arr >= 0) & (sample_arr < nvec))
        
        else:
            valid = False
        
        results.append(valid)
    
    return results