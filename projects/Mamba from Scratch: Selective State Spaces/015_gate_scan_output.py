def gate_scan_output(y, z):
    """Modulate the selective-scan output y by the parallel gate branch z."""
    return y * silu(z)