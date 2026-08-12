def pooled_stats(m1, s1, n1, m2, s2, n2):
    N = n1 + n2
    combined_mean = (n1 * m1 + n2 * m2) / N
    ss_within = (n1 - 1) * s1**2 + (n2 - 1) * s2**2
    ss_between = n1 * (m1 - combined_mean)**2 + n2 * (m2 - combined_mean)**2
    ss_total = ss_within + ss_between
    combined_var = ss_total / (N - 1)
    combined_sd = combined_var ** 0.5
    return combined_mean, combined_sd
    
