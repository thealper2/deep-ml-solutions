def partition_parameters(params):
    """
    Partition parameter names into Muon and AdamW groups.

    Args:
        params: list of (name: str, shape: tuple[int, ...]) descriptors.

    Returns:
        dict with keys 'muon' and 'adamw', each a list of names preserving
        the original order from `params`.
    """
    muon = []
    adamw = []

    for param in params:
        if len(param[1]) == 2 and 'embed' not in param[0]:
            muon.append(param[0])
        else:
            adamw.append(param[0])

    return {
        'muon': muon,
        'adamw': adamw,
    }