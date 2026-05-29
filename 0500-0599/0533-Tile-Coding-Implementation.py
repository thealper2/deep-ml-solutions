import numpy as np

def tile_coding(
    state: list[float],
    num_tilings: int,
    tiles_per_dim: list[int],
    state_low: list[float],
    state_high: list[float]
) -> tuple[list[int], int]:
    """
    Compute active tile indices using tile coding.
    
    Args:
        state: Continuous state values (one per dimension)
        num_tilings: Number of overlapping tilings
        tiles_per_dim: Number of tiles per dimension in each tiling
        state_low: Lower bounds for each dimension
        state_high: Upper bounds for each dimension
        
    Returns:
        Tuple of (active_tile_indices, total_number_of_tiles)
    """
    state = np.array(state)
    tiles_per_dim = np.array(tiles_per_dim)
    state_low = np.array(state_low)
    state_high = np.array(state_high)

    d = len(state)
    tile_widths = (state_high - state_low) / tiles_per_dim

    tiles_per_tiling = int(np.prod(tiles_per_dim))
    total_tiles = num_tilings * tiles_per_tiling

    active_indices = []

    for t in range(num_tilings):
        offsets = t * (tile_widths / num_tilings)
        adjusted = state + offsets - state_low
        tile_coords = np.floor(adjusted / tile_widths).astype(int)
        tile_coords = np.clip(tile_coords, 0, tiles_per_dim - 1)

        idx, multiplier = 0, 1
        for dim in range(d - 1, -1, -1):
            idx += tile_coords[dim] * multiplier
            multiplier *= tiles_per_dim[dim]

        global_idx = t * tiles_per_tiling + idx
        active_indices.append(global_idx)

    return active_indices, total_tiles