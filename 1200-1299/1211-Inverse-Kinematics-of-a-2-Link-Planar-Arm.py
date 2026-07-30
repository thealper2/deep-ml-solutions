import math

def inverse_kinematics(x, y, l1, l2, elbow='up'):
    r = math.sqrt(x**2 + y**2)
    
    if r < abs(l1 - l2) or r > l1 + l2:
        return None
    
    phi = math.atan2(y, x)
    
    D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    D = max(-1.0, min(1.0, D))
    theta2_abs = math.acos(D)
    
    cos_beta = (r**2 + l1**2 - l2**2) / (2 * r * l1)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = math.acos(cos_beta)
    
    if elbow == 'up':
        theta1 = phi + beta
        theta2 = -theta2_abs
    elif elbow == 'down':
        theta1 = phi - beta
        theta2 = theta2_abs
    
    return [theta1, theta2]
