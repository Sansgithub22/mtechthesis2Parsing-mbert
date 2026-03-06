"""
Training loop for the biaffine dependency parser.

Features:
    - Inverse square root learning rate schedule with linear warmup
      (following the paper's setup)
    - Separate learning rate for BERT vs parser parameters
    - Early stopping based on validation LAS
    - Checkpoint saving (best model only)
    - Full training log saved to a JSON file for analysis
"""

import os
import json
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional
from tqdm import tqdm

from data.conllu_reader import Token
from data.dataset import DependencyDataset, collate_fn
from models.biaffine_parser import BiaffineParser
from evaluation.metrics import ParseEvaluator


class InvSqrtScheduler:
    """
    Inverse square root learning rate schedule with linear warmup.
    lr = base_lr * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, optimizer, warmup_steps: int = 160):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr_scale = self._get_scale()
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["base_lr"] * lr_scale

    def _get_scale(self):
        s = self._step
        w = self.warmup_steps
        return min(s ** -0.5, s * w ** -1.5) * (w ** 0.5)


def build_optimizer(model: BiaffineParser, bert_lr: float, parser_lr: float):
    """
    Build Adam optimizer with two parameter groups:
      - BERT encoder: lower learning rate (bert_lr)
      - Parser head (MLPs, biaffine layers, BiLSTM): higher learning rate (parser_lr)
    """
    bert_params = list(model.encoder.parameters())
    bert_ids = set(id(p) for p in bert_params)
    parser_params = [p for p in model.parameters() if id(p) not in bert_ids]

    param_groups = [
        {"params": bert_params, "base_lr": bert_lr, "lr": bert_lr},
        {"params": parser_params, "base_lr": parser_lr, "lr": parser_lr},
    ]
    return torch.optim.Adam(param_groups, betas=(0.9, 0.9))


class ParserTrainer:
    """
    Trains and evaluates a BiaffineParser.

    Args:
        model: BiaffineParser instance
        train_sentences: training split (list of sentences)
        dev_sentences: validation split
        tokenizer: BERT tokenizer
        rel_vocab: dict mapping relation string -> int
        save_dir: directory to save best checkpoint and logs
        batch_size: number of sentences per batch
        max_epochs: maximum training epochs
        patience: early stopping patience (epochs without improvement)
        bert_lr: learning rate for BERT parameters
        parser_lr: learning rate for parser head parameters
        warmup_epochs: number of epochs for LR warmup (approximate)
        max_grad_norm: gradient clipping threshold
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model: BiaffineParser,
        train_sentences: List[List[Token]],
        dev_sentences: List[List[Token]],
        tokenizer,
        rel_vocab: dict,
        save_dir: str,
        batch_size: int = 24,
        max_epochs: int = 200,
        patience: int = 20,
        bert_lr: float = 5e-5,
        parser_lr: float = 1e-3,
        warmup_epochs: int = 1,
        max_grad_norm: float = 5.0,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.rel_vocab = rel_vocab
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.device = device

        os.makedirs(save_dir, exist_ok=True)

        # Build datasets and dataloaders
        pad_id = tokenizer.pad_token_id

        self.train_dataset = DependencyDataset(train_sentences, tokenizer, rel_vocab)
        self.dev_dataset = DependencyDataset(dev_sentences, tokenizer, rel_vocab)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )
        self.dev_loader = DataLoader(
            self.dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

        self.optimizer = build_optimizer(model, bert_lr, parser_lr)

        # Approximate warmup in steps
        warmup_steps = warmup_epochs * max(1, len(self.train_loader))
        self.scheduler = InvSqrtScheduler(self.optimizer, warmup_steps)

        # Keep dev sentences for evaluation (evaluator needs Token objects)
        self.dev_sentences = dev_sentences
        self.evaluator = ParseEvaluator()
        self.log = []

    def train(self):
        """Run the full training loop. Returns the best validation LAS."""
        best_las = 0.0
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            las, uas = self._evaluate()
            elapsed = time.time() - t0

            entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "dev_las": round(las, 2),
                "dev_uas": round(uas, 2),
                "time_s": round(elapsed, 1),
            }
            self.log.append(entry)
            print(
                f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"LAS={las:.2f} UAS={uas:.2f} | {elapsed:.1f}s"
            )

            if las > best_las:
                best_las = las
                no_improve = 0
                self._save_checkpoint("best_model.pt")
                print(f"  => New best LAS: {best_las:.2f}")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        self._save_log()
        print(f"\nTraining complete. Best validation LAS: {best_las:.2f}")
        return best_las

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            _, _, loss = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_starts=batch["word_starts"],
                word_mask=batch["word_mask"],
                heads=batch["heads"],
                rels=batch["rels"],
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self) -> tuple:
        self.model.eval()
        self.evaluator.reset()

        # Map dev_sentences to their dataset index order
        # (some sentences may have been filtered out if too long)
        dev_sentence_list = self.dev_sentences

        idx = 0
        for batch in self.dev_loader:
            batch_size = batch["input_ids"].size(0)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            pred_heads, pred_rels = self.model.predict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_starts=batch["word_starts"],
                word_mask=batch["word_mask"],
            )

            # Match back to original sentences
            batch_sents = dev_sentence_list[idx: idx + batch_size]
            self.evaluator.update(batch_sents, pred_heads, pred_rels, self.rel_vocab)
            idx += batch_size

        return self.evaluator.get_scores()

    def _save_checkpoint(self, name: str):
        path = os.path.join(self.save_dir, name)
        torch.save(self.model.state_dict(), path)

    def _save_log(self):
        path = os.path.join(self.save_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)

    def load_best(self):
        """Load the best checkpoint saved during training."""
        path = os.path.join(self.save_dir, "best_model.pt")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def evaluate_test(self, test_sentences: List[List[Token]], tokenizer) -> tuple:
        """Evaluate on the test set using the best saved model."""
        self.load_best()
        pad_id = tokenizer.pad_token_id
        test_dataset = DependencyDataset(test_sentences, tokenizer, self.rel_vocab)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

        self.model.eval()
        evaluator = ParseEvaluator()
        idx = 0

        with torch.no_grad():
            for batch in test_loader:
                bs = batch["input_ids"].size(0)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                pred_heads, pred_rels = self.model.predict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    word_starts=batch["word_starts"],
                    word_mask=batch["word_mask"],
                )

                batch_sents = test_sentences[idx: idx + bs]
                evaluator.update(batch_sents, pred_heads, pred_rels, self.rel_vocab)
                idx += bs

        las, uas = evaluator.get_scores()
        print(f"Test LAS={las:.2f}  UAS={uas:.2f}  (over {evaluator.total} tokens)")
        return las, uas
