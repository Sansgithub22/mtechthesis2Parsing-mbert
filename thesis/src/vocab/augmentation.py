"""
Vocabulary Augmentation (VA) for mBERT/XLM-R.

The paper's idea:
    mBERT has 99 unused token slots in its vocabulary.
    We train a new wordpiece vocabulary on the target language.
    We find wordpieces in the new vocabulary that reduce unknown tokens
    compared to the original mBERT vocabulary.
    We inject the 99 most frequent such wordpieces into mBERT's unused slots.

After injection, we run LAPT with the augmented vocabulary so the new
embeddings get trained.

This module:
    1. Trains a new wordpiece vocabulary on target language text
    2. Identifies beneficial novel wordpieces
    3. Injects them into mBERT's tokenizer + model
    4. Saves the augmented model for use in VA+LAPT
"""

import os
import json
import copy
from collections import Counter
from typing import List, Tuple

from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer


def train_wordpiece_vocab(
    text_path: str,
    output_vocab_path: str,
    vocab_size: int = 5000,
    min_frequency: int = 2,
    lowercase: bool = False,
) -> str:
    """
    Train a new wordpiece vocabulary on target language text.

    Args:
        text_path: path to training text (one sentence per line)
        output_vocab_path: where to save the new vocab.txt
        vocab_size: total vocabulary size
        min_frequency: minimum token frequency to include in vocab
        lowercase: whether to lowercase the text

    Returns:
        output_vocab_path
    """
    print(f"Training wordpiece vocab from {text_path} ...")

    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)
    tokenizer.train(
        files=[text_path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    os.makedirs(os.path.dirname(output_vocab_path) or ".", exist_ok=True)
    tokenizer.save_model(os.path.dirname(output_vocab_path) or ".")

    # The tokenizer saves to vocab.txt; rename if needed
    saved_vocab = os.path.join(os.path.dirname(output_vocab_path) or ".", "vocab.txt")
    if saved_vocab != output_vocab_path:
        os.rename(saved_vocab, output_vocab_path)

    print(f"New vocab saved to: {output_vocab_path}")
    return output_vocab_path


def find_novel_wordpieces(
    text_path: str,
    original_tokenizer,
    new_vocab_path: str,
    top_k: int = 99,
) -> List[Tuple[str, int]]:
    """
    Find wordpieces in the new vocabulary that reduce unknown tokens.

    Algorithm:
        For each word in the corpus:
            - Tokenize with original mBERT vocab
            - Tokenize with new vocab
            - If new vocab produces fewer [UNK] tokens, note the novel pieces
        Count how often each novel piece appears across the corpus.
        Return the top_k most frequent novel pieces.

    Args:
        text_path: path to target language text
        original_tokenizer: loaded mBERT tokenizer
        new_vocab_path: path to the new vocab.txt file
        top_k: how many novel pieces to return

    Returns:
        List of (wordpiece, frequency) tuples, sorted by frequency descending
    """
    print("Finding novel wordpieces ...")

    # Load the new vocabulary as a set for quick lookup
    with open(new_vocab_path, encoding="utf-8") as f:
        new_vocab = set(line.strip() for line in f)

    # Words in original mBERT vocab
    original_vocab = set(original_tokenizer.vocab.keys())

    # Count novel wordpieces that fix unknown tokens
    novel_counts = Counter()

    with open(text_path, encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                orig_pieces = original_tokenizer.tokenize(word)
                if "[UNK]" not in orig_pieces:
                    continue  # original handles this word fine

                # New vocab tokenizes this word better — find the novel pieces
                # We do a simple check: which pieces in new_vocab are not in original_vocab
                for piece in _wordpiece_tokenize(word, new_vocab):
                    if piece not in original_vocab and piece != "[UNK]":
                        novel_counts[piece] += 1

    top_pieces = novel_counts.most_common(top_k)
    print(f"Found {len(novel_counts)} novel pieces; top {top_k}: {top_pieces[:5]}...")
    return top_pieces


def _wordpiece_tokenize(word: str, vocab: set) -> List[str]:
    """
    Simple greedy wordpiece tokenization using a given vocabulary set.
    This mirrors the algorithm used by BERT's tokenizer.
    """
    tokens = []
    start = 0
    while start < len(word):
        end = len(word)
        found = None
        while start < end:
            substr = word[start:end]
            candidate = substr if start == 0 else "##" + substr
            if candidate in vocab:
                found = candidate
                break
            end -= 1
        if found is None:
            tokens.append("[UNK]")
            break
        tokens.append(found)
        start = end
    return tokens


def inject_vocab_into_mbert(
    model_name_or_path: str,
    novel_pieces: List[Tuple[str, int]],
    output_dir: str,
) -> str:
    """
    Inject novel wordpieces into mBERT's unused token slots.

    mBERT has exactly 99 tokens named [unused0] .. [unused98].
    We replace these with our novel wordpieces.
    The model's embedding matrix is kept as-is — the new embeddings
    start from the unused token's learned representations (near-zero),
    which is fine since they will be trained during LAPT.

    Args:
        model_name_or_path: original mBERT or checkpoint path
        novel_pieces: list of (wordpiece, freq) from find_novel_wordpieces()
        output_dir: where to save the augmented model and tokenizer

    Returns:
        output_dir
    """
    print(f"Injecting {len(novel_pieces)} novel pieces into tokenizer ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

    # Find unused token IDs in the vocabulary
    unused_tokens = [
        tok for tok in tokenizer.vocab
        if tok.startswith("[unused")
    ]
    unused_tokens.sort(key=lambda t: int(t[7:-1]))  # sort by index

    if len(unused_tokens) < len(novel_pieces):
        print(f"WARNING: Only {len(unused_tokens)} unused slots, "
              f"but {len(novel_pieces)} novel pieces requested. "
              f"Will inject {len(unused_tokens)}.")
        novel_pieces = novel_pieces[:len(unused_tokens)]

    # The tokenizer vocab is stored as a dict. We need to modify it.
    # The cleanest way is to rebuild the vocab.txt file.
    vocab = copy.copy(tokenizer.vocab)  # word -> id

    injected = []
    for i, (piece, freq) in enumerate(novel_pieces):
        if i >= len(unused_tokens):
            break
        old_token = unused_tokens[i]
        old_id = vocab[old_token]
        # Replace old_token with new piece at same ID
        del vocab[old_token]
        vocab[piece] = old_id
        injected.append((piece, old_id))

    print(f"Injected {len(injected)} wordpieces")

    # Save the new vocabulary to output_dir
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.txt")

    # vocab.txt: one token per line, ordered by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")

    # Copy other tokenizer files and reload with new vocab
    tokenizer.save_pretrained(output_dir)
    # Overwrite vocab.txt with our modified version
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")

    # Save the model (embedding matrix unchanged)
    model.save_pretrained(output_dir)

    print(f"Augmented model saved to: {output_dir}")
    return output_dir


def run_va_pipeline(
    base_model: str,
    text_path: str,
    eval_path: str,
    output_dir: str,
    vocab_size: int = 5000,
    top_k: int = 99,
    num_lapt_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    fp16: bool = True,
) -> str:
    """
    Full VA pipeline:
        1. Train new wordpiece vocab
        2. Find novel wordpieces
        3. Inject into mBERT
        4. Run LAPT with augmented vocab

    Args:
        base_model: mBERT model ID
        text_path: unlabeled target language text
        eval_path: validation text
        output_dir: base output directory
        vocab_size: size of new wordpiece vocab
        top_k: number of novel pieces to inject
        num_lapt_epochs: LAPT epochs after injection
        batch_size: LAPT batch size
        learning_rate: LAPT learning rate
        fp16: use mixed precision

    Returns:
        Path to the final VA+LAPT model
    """
    from training.lapt import run_lapt

    # Step 1: Train new vocab
    new_vocab_dir = os.path.join(output_dir, "new_vocab")
    os.makedirs(new_vocab_dir, exist_ok=True)
    new_vocab_path = os.path.join(new_vocab_dir, "vocab.txt")
    train_wordpiece_vocab(text_path, new_vocab_path, vocab_size=vocab_size)

    # Step 2: Find novel wordpieces
    original_tokenizer = AutoTokenizer.from_pretrained(base_model)
    novel_pieces = find_novel_wordpieces(text_path, original_tokenizer, new_vocab_path, top_k=top_k)

    if not novel_pieces:
        print("No novel wordpieces found! Falling back to standard LAPT.")
        augmented_dir = base_model
    else:
        # Step 3: Inject into mBERT
        augmented_dir = os.path.join(output_dir, "augmented_model")
        inject_vocab_into_mbert(base_model, novel_pieces, augmented_dir)

    # Step 4: LAPT on augmented model
    lapt_output = os.path.join(output_dir, "va_lapt")
    run_lapt(
        model_name_or_path=augmented_dir,
        train_text_path=text_path,
        eval_text_path=eval_path,
        output_dir=lapt_output,
        num_epochs=num_lapt_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=fp16,
    )

    return lapt_output
