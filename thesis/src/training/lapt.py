"""
Language-Adaptive Pre-Training (LAPT).

This module re-trains mBERT (or XLM-R) on a target language's unlabeled text
using the Masked Language Modeling (MLM) objective. This is the core idea of
the paper: treat low-resource language adaptation like domain adaptation.

We use the HuggingFace Trainer which handles:
    - Gradient accumulation
    - Mixed precision (fp16) on GPU
    - Checkpoint saving
    - Evaluation on a held-out set

Reference:
    Section 2.1 of Chau et al. (2020), and
    Gururangan et al. (2020) "Don't Stop Pretraining"
"""

import os
import math
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


def load_text_file(path: str) -> list:
    """Load a plain text file and return a list of non-empty lines."""
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def run_lapt(
    model_name_or_path: str,
    train_text_path: str,
    eval_text_path: str,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    mlm_probability: float = 0.15,
    max_length: int = 128,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
) -> str:
    """
    Run Language-Adaptive Pre-Training (LAPT).

    Loads a pre-trained multilingual model and continues training it on
    target-language text using MLM loss only (no NSP, as in the paper).

    Args:
        model_name_or_path: HuggingFace model ID or local checkpoint path.
            Use 'bert-base-multilingual-cased' for mBERT, or
            'xlm-roberta-base' for XLM-R.
        train_text_path: path to training text file (one sentence per line)
        eval_text_path: path to evaluation text file (one sentence per line)
        output_dir: directory to save the LAPT-adapted model
        num_epochs: number of pretraining epochs
        batch_size: per-device batch size
        learning_rate: learning rate for AdamW
        warmup_steps: linear warmup steps
        mlm_probability: probability of masking each token
        max_length: maximum sequence length in wordpieces
        fp16: use mixed-precision training (requires GPU)
        gradient_accumulation_steps: accumulate gradients over N steps

    Returns:
        output_dir: path to the saved model (for use in downstream training)
    """
    print(f"\n{'='*60}")
    print(f"Running LAPT: {model_name_or_path}")
    print(f"  Train data: {train_text_path}")
    print(f"  Epochs: {num_epochs}  |  LR: {learning_rate}  |  Batch: {batch_size}")
    print(f"{'='*60}\n")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

    # Load text data into HuggingFace Dataset format
    train_texts = load_text_file(train_text_path)
    eval_texts = load_text_file(eval_text_path)

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Dynamic masking collator (masks a different set of tokens each time)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    # Compute total training steps for logging
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size / gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=fp16 and torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to="none",       # disable wandb/tensorboard by default
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the final model (best checkpoint is already loaded due to load_best_model_at_end)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nLAPT complete. Model saved to: {output_dir}")
    return output_dir


def run_lapt_for_all_languages(
    base_model: str,
    data_root: str,
    output_root: str,
    language_epochs: dict,
    **kwargs,
):
    """
    Convenience function: run LAPT for all four languages.

    Args:
        base_model: 'bert-base-multilingual-cased' or 'xlm-roberta-base'
        data_root: directory containing unlabeled data (e.g., data/unlabeled/)
        output_root: base directory for saving LAPT models
        language_epochs: dict mapping language code to number of epochs
            e.g., {'ga': 5, 'mt': 20, 'sing': 5, 'vi': 5}
        **kwargs: additional arguments passed to run_lapt()

    Returns:
        dict mapping language code -> path to LAPT model
    """
    lang_map = {
        "ga": "ga",      # Irish
        "mt": "mt",      # Maltese
        "sing": "sg",    # Singlish (data dir is 'sg')
        "vi": "vi",      # Vietnamese
    }

    results = {}
    for lang, data_dir in lang_map.items():
        epochs = language_epochs.get(lang, 5)
        train_path = os.path.join(data_root, data_dir, "train.txt")
        eval_path = os.path.join(data_root, data_dir, "valid.txt")
        out_dir = os.path.join(output_root, f"lapt_{lang}")

        if not os.path.exists(train_path):
            print(f"WARNING: {train_path} not found, skipping {lang}")
            continue

        results[lang] = run_lapt(
            model_name_or_path=base_model,
            train_text_path=train_path,
            eval_text_path=eval_path,
            output_dir=out_dir,
            num_epochs=epochs,
            **kwargs,
        )

    return results
