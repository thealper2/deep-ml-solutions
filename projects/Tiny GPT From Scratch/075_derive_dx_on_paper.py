def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    notes = (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )
    return notes
