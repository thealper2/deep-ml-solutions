import numpy as np
import math

def slerp(q0, q1, t):
    q0 = [float(x) for x in q0]
    q1 = [float(x) for x in q1]
    dot = sum(q0[i] * q1[i] for i in range(4))

    if dot < 0:
        q1 = [-x for x in q1]
        dot = -dot

    if dot > 0.9995:
        result = [q0[i] + t * (q1[i] - q0[i]) for i in range(4)]
        norm = np.sqrt(sum(x * x for x in result))
        return [x / norm for x in result]

    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    w0 = math.sin((1 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta

    result = [w0 * q0[i] + w1 * q1[i] for i in range(4)]
    norm = np.sqrt(sum(x * x for x in result))
    return [x / norm for x in result]
