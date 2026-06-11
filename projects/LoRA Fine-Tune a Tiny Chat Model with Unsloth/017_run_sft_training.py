def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    result = trainer.train()
    return float(result.training_loss)
