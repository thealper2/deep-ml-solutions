def compute_non_iid_gap(iid_accuracies, non_iid_accuracies):
    iid_final = iid_accuracies[-1]
    non_iid_final = non_iid_accuracies[-1]
    gap = iid_final - non_iid_final
    return {
        'iid_final': float(iid_final),
        'non_iid_final': float(non_iid_final),
        'gap': float(gap)
    }
