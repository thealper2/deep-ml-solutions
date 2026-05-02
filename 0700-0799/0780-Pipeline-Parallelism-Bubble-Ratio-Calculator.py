def pipeline_bubble_ratio(PP: int, V: int, M: int) -> float:
    """
    Compute the pipeline parallelism bubble ratio.

    Args:
        PP: number of pipeline stages
        V: number of virtual stages per pipeline rank
        M: number of micro-batches

    Returns:
        bubble ratio as a float
    """
    bubble_ratio = (PP - 1) / (V * M)
    return round(bubble_ratio, 4)