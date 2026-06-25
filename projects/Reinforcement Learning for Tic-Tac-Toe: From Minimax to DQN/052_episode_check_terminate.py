def episode_check_terminate(status):
    """Return True if status is terminal (win or draw), else False."""
    return status != 'ongoing'
