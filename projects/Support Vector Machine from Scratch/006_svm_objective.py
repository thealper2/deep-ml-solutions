def svm_objective(x, y, params, reg_lambda):
    scores = compute_scores(x, params)
    mean_hinge = np.mean([hinge_loss_example(score, yi) for score, yi in zip(scores, y)])
    reg = reg_lambda * np.sum(params['w'] ** 2)
    return mean_hinge + reg
