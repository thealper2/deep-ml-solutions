def string_kernel(s: str, t: str, p: int) -> int:
    """
    Compute the p-spectrum string kernel between two strings.
    """
    if len(s) < p or len(t) < p:
        return 0
    
    count_s = {}
    for i in range(len(s) - p + 1):
        sub = s[i:i+p]
        count_s[sub] = count_s.get(sub, 0) + 1
    
    kernel = 0
    for i in range(len(t) - p + 1):
        sub = t[i:i+p]
        kernel += count_s.get(sub, 0)
    
    return kernel