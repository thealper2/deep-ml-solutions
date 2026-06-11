"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_base_model_and_tokenizer ──
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, 
        max_seq_length=max_seq_length
    )
    return model, tokenizer

# ── Step 002  count_total_parameters ──
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# ── Step 003  is_model_4bit_quantized ──
import torch
import bitsandbytes as bnb

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            return True

    return False

# ── Step 004  ensure_pad_token ──
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# ── Step 005  get_lora_target_modules ──
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# ── Step 006  attach_lora_adapters ──
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    if target_modules is None:
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias='none',
    )

    return model

# ── Step 007  count_trainable_parameters ──
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params

# ── Step 008  trainable_fraction ──
def trainable_fraction(trainable_count, total_count):
    return trainable_count / total_count if total_count > 0 else 0.0

# ── Step 009  build_instruction_examples ──
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    examples = []
    for i in range(3):
        examples.append({'instruction': f'What is 3 - {i}', 'response': f'Result is {3 - i}'})

    return examples

# ── Step 010  format_instruction_example ──
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# ── Step 011  format_all_examples ──
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}" for example in examples]

# ── Step 012  build_text_dataset ──
from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    data = {'text': texts}
    return Dataset.from_dict(data)

# ── Step 013  tokenize_text ──
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    return token_ids

# ── Step 014  count_tokens ──
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    return len(input_ids)

# ── Step 015  build_training_arguments ──
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
        save_strategy="no",
        save_only_model=False,
        ddp_find_unused_parameters=False,
    )

# ── Step 016  build_sft_trainer ──
from trl import SFTTrainer

def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

# ── Step 017  run_sft_training ──
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    result = trainer.train()
    return float(result.training_loss)

# ── Step 018  switch_to_inference_mode ──
from unsloth import FastLanguageModel

def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    return FastLanguageModel.for_inference(model)

# ── Step 019  build_chat_prompt ──
def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

# ── Step 020  generate_reply ──
def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    outputs =model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    prompt_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0, prompt_length:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply

# ── Scaffold (runner) ──
"""Scaffold: LoRA fine-tune a tiny 4-bit Qwen2.5 chat model with Unsloth."""
import torch

from solution import (
    load_base_model_and_tokenizer,
    count_total_parameters,
    is_model_4bit_quantized,
    ensure_pad_token,
    get_lora_target_modules,
    attach_lora_adapters,
    count_trainable_parameters,
    trainable_fraction,
    build_instruction_examples,
    format_instruction_example,
    format_all_examples,
    build_text_dataset,
    tokenize_text,
    count_tokens,
    build_training_arguments,
    build_sft_trainer,
    run_sft_training,
    switch_to_inference_mode,
    build_chat_prompt,
    generate_reply,
)


def main():
    torch.manual_seed(0)

    # 1) Load 4-bit base model + tokenizer.
    model, tokenizer = load_base_model_and_tokenizer(
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=256,
    )
    total_params = count_total_parameters(model)
    quantized = is_model_4bit_quantized(model)
    print(f"[base] total_params={total_params:,} 4bit={quantized}")

    # 2) Make sure tokenizer has a pad token.
    tokenizer = ensure_pad_token(tokenizer)
    print(f"[tokenizer] pad_token={tokenizer.pad_token!r}")

    # 3) Attach LoRA adapters to attention projections.
    target_modules = get_lora_target_modules()
    print(f"[lora] target_modules={target_modules}")
    model = attach_lora_adapters(
        model, r=8, lora_alpha=16, target_modules=target_modules
    )
    trainable = count_trainable_parameters(model)
    frac = trainable_fraction(trainable, total_params)
    print(f"[lora] trainable={trainable:,} fraction={frac:.6f}")

    # 4) Build a tiny in-code instruction dataset.
    examples = build_instruction_examples()
    print(f"[data] num_examples={len(examples)}")
    first_text = format_instruction_example(examples[0])
    print(f"[data] formatted[0]={first_text!r}")
    texts = format_all_examples(examples)
    dataset = build_text_dataset(texts)
    print(f"[data] dataset_columns={dataset.column_names} size={len(dataset)}")

    # 5) Peek at tokenization of one example.
    ids = tokenize_text(tokenizer, texts[0])
    print(f"[data] tokens[0]={count_tokens(ids)}")

    # 6) Featherweight SFT: 5 steps, batch size 1.
    training_args = build_training_arguments(
        output_dir="./sft_out", max_steps=5, learning_rate=2e-4
    )
    trainer = build_sft_trainer(
        model, tokenizer, dataset, training_args, max_seq_length=256
    )
    final_loss = run_sft_training(trainer)
    print(f"[train] final_loss={final_loss:.4f}")

    # 7) Switch to fast inference and generate a reply.
    switch_to_inference_mode(model)
    prompt = build_chat_prompt(tokenizer, "Say hello in one short sentence.")
    reply = generate_reply(model, tokenizer, prompt, max_new_tokens=32)
    print(f"[gen] reply={reply!r}")

    passed = (
        total_params > 0
        and trainable > 0
        and trainable < total_params
        and 0.0 < frac < 0.1
        and isinstance(reply, str)
        and len(reply) > 0
        and final_loss == final_loss  # finite (not NaN)
    )
    print({"passed": bool(passed)})
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
