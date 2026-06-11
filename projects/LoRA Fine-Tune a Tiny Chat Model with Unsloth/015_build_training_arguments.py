from transformers import TrainingArguments
import torch

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=1,
        optim='adamw_8bit',
    )
