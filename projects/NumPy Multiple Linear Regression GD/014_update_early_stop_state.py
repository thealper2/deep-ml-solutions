def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    if val_loss < best_val_loss:
        return val_loss, 0, weights.copy(), False
    else:
        wait += 1
        if wait >= patience:
            return best_val_loss, wait, best_weights, True

        return best_val_loss, wait, best_weights, False
