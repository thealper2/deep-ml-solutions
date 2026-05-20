def distribute_attention_layers(n_layers: int, ratio: str) -> dict:
    """
    Distribute n_layers into SWA and FULL attention layers following the
    given ratio 'a:b' (SWA:Full), using a repeating cycle pattern.
    Return a dict with keys 'swa', 'full', and 'layers'.
    """
    swa_ratio, full_ratio = map(int, ratio.split(':'))
    cycle_length = swa_ratio + full_ratio
    
    layers = []
    for i in range(n_layers):
        pos_in_cycle = i % cycle_length
        if pos_in_cycle < swa_ratio:
            layers.append('SWA')
        else:
            layers.append('FULL')
    
    return {
        'swa': layers.count('SWA'),
        'full': layers.count('FULL'),
        'layers': layers,
    }