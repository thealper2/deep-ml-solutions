import numpy as np

def tile_code(position, velocity, n_tilings, tiles_per_dim):
    pos_scaled = (position + 1.2) / 1.7 * tiles_per_dim
    vel_scaled = (velocity + 0.07) / 0.14 * tiles_per_dim

    width = tiles_per_dim + 1
    out = np.empty(n_tilings, dtype=int)
    for i in range(n_tilings):
        offset = i / n_tilings
        px = min(int(np.floor(pos_scaled + offset)), tiles_per_dim)
        vx = min(int(np.floor(vel_scaled + offset)), tiles_per_dim)
        out[i] = i * width * width + px * width + vx
        
    return out