from unsloth import FastLanguageModel

def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    return FastLanguageModel.for_inference(model)
