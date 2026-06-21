def assign_value_targets(history, winner):
    labelled = []
    for step in history:
        new_step = step.copy()
        to_play = step['to_play']
        if winner == 0:
            new_step['value'] = 0.0
        elif to_play == winner:
            new_step['value'] = 1.0
        else:
            new_step['value'] = -1.0

        labelled.append(new_step)

    return labelled
