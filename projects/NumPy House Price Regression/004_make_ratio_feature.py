def make_ratio_feature(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)
