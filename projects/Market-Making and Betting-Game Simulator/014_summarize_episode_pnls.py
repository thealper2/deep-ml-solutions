def summarize_episode_pnls(pnls):
    pnls = np.array(pnls)
    return {
        'mean': float(np.mean(pnls)),
        'std': float(np.std(pnls)),
        'worst': float(np.min(pnls))
    }
