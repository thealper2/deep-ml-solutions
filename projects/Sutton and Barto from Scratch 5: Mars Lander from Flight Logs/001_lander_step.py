import numpy as np

MAX_ALTITUDE = 16
MAX_SPEED = 5
MAX_FUEL = 24
SAFE_SPEED = 2
START = (16, 0, 24)


def lander_step(state, action, rng):
    h, v, f = state
    thrust = min(action, f)
    new_fuel = f - thrust

    u = rng.random()
    if u < 0.1:
        gust = -1
    elif u >= 0.9:
        gust = 1
    else:
        gust = 0

    new_v = v + thrust - 1 + gust
    new_v = max(-MAX_SPEED, min(MAX_SPEED, new_v))
    new_v = int(new_v)

    new_h = h + new_v

    if new_h <= 0:
        reward = 1.0 if new_v >= -SAFE_SPEED else 0.0
        return ((0, new_v, int(new_fuel)), reward, True)

    return ((int(min(new_h, MAX_ALTITUDE)), new_v, int(new_fuel)), 0.0, False)