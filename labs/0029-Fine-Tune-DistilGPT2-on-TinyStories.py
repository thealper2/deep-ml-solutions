import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

def train(model, tokenizer, train_texts, val_texts):
    """Fine-tune `model` on `train_texts` (TinyStories samples).

    Any method works — full fine-tuning, LoRA via `peft`, prefix tuning,
    layer freezing, etc. The validator measures val_loss after your
    training and reports it — you don't need to print anything.

    Args:
        model:        AutoModelForCausalLM (distilgpt2), cuda + fp16.
        tokenizer:    matching AutoTokenizer.
        train_texts:  list[str]  -- TinyStories train samples
        val_texts:    list[str]  -- TinyStories val samples

    Returns:
        The trained model.
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    def tokenize_texts(texts, max_length=256):
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            return self.texts[idx]
    
    train_dataset = TextDataset(train_texts)
    val_dataset = TextDataset(val_texts)
    
    def collate_fn(batch):
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["input_ids"]
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item() * len(batch["input_ids"])
    
    return model
