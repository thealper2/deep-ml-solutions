import math
import numpy as np

def mountain_car_step(position, velocity, action):
    velocity = velocity + 0.001 * (action - 1) - 0.0025 * math.cos(3.0 * position)
    velocity = max(-0.07, min(0.07, velocity))

    position = position + velocity
    position = max(-1.2, min(0.5, position))

    if position <= -1.2:
        velocity = 0.0

    done = bool(position >= 0.5)
    reward = -1.0

    return (float(position), float(velocity), reward, done)