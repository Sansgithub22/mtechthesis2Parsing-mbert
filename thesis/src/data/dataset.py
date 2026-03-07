"""
PyTorch Dataset for dependency parsing with BERT-based encoders.

Key challenge: BERT tokenizes words into multiple wordpieces.
We need to align wordpiece indices back to word indices for parsing.

Strategy: for each word, keep only the FIRST wordpiece's representation.
This follows common practice (e.g., Kondratyuk & Straka, 2019).
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from transformers import PreTrainedTokenizerFast
from data.conllu_reader import Token


class DependencyDataset(Dataset):
    """
    Dataset for dependency parsing.

    Each item contains:
        input_ids: [seq_len] wordpiece token IDs (with [CLS] and [SEP])
        attention_mask: [seq_len] 1 for real tokens, 0 for padding
        word_starts: [n_words] indices in input_ids of first wordpiece for each word
        heads: [n_words] gold head index for each word (0 = root)
        rels: [n_words] gold relation index for each word
    """

    def __init__(
        self,
        sentences: List[List[Token]],
        tokenizer: PreTrainedTokenizerFast,
        rel_vocab: Dict[str, int],
        max_length: int = 128,
    ):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.rel_vocab = rel_vocab
        self.max_length = max_length
        self.items = [self._encode(s) for s in sentences]
        # Filter out None (sentences that are too long after tokenization)
        self.items = [x for x in self.items if x is not None]

    def _encode(self, sentence: List[Token]) -> Optional[dict]:
        """
        Tokenize a sentence and build alignment between wordpieces and words.
        Returns None if sentence is too long.

        A virtual ROOT token is prepended at position 0 (mapped to [CLS]).
        UD head values (0=root, 1..N=1-indexed words) then correctly index
        into tensor positions 0..N, eliminating out-of-bounds errors.
        """
        words = [t.form for t in sentence]
        heads_raw = [t.head for t in sentence]   # UD: 0=root, 1..N (1-indexed)
        rels_raw = [self.rel_vocab.get(t.deprel, 0) for t in sentence]

        # Tokenize each word separately to track which wordpieces belong to which word
        all_pieces = []
        word_starts_real = []  # index of first wordpiece for each word (offset by 1 for [CLS])

        for word in words:
            pieces = self.tokenizer.tokenize(word)
            if not pieces:
                pieces = [self.tokenizer.unk_token]
            word_starts_real.append(len(all_pieces) + 1)  # +1 for [CLS]
            all_pieces.extend(pieces)

        # Build full sequence: [CLS] + all_pieces + [SEP]
        total_len = len(all_pieces) + 2  # [CLS] and [SEP]

        if total_len > self.max_length:
            return None

        # Convert to token IDs
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        piece_ids = self.tokenizer.convert_tokens_to_ids(all_pieces)

        input_ids = [cls_id] + piece_ids + [sep_id]
        attention_mask = [1] * len(input_ids)

        # Prepend virtual ROOT at position 0 (mapped to [CLS] token at index 0).
        # Root's own head/rel are padding values — they are masked out in loss/eval.
        word_starts = [0] + word_starts_real
        heads = [0] + heads_raw
        rels = [0] + rels_raw
        # is_real_word: 0 for root, 1 for actual words (controls loss/eval mask)
        is_real_word = [0] + [1] * len(sentence)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_starts": word_starts,
            "heads": heads,
            "rels": rels,
            "is_real_word": is_real_word,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch: List[dict], pad_token_id: int = 0) -> dict:
    """
    Collate a batch of variable-length examples into padded tensors.

    Args:
        batch: list of dicts from DependencyDataset.__getitem__
        pad_token_id: token ID to use for padding input_ids

    Returns:
        dict of tensors, each with shape [batch_size, max_len]
    """
    max_seq = max(len(x["input_ids"]) for x in batch)
    max_words = max(len(x["heads"]) for x in batch)

    input_ids_padded = []
    attention_mask_padded = []
    word_starts_padded = []
    heads_padded = []
    rels_padded = []
    word_mask = []  # 1 for real words, 0 for padding

    for x in batch:
        seq_len = len(x["input_ids"])
        n_words = len(x["heads"])  # includes root at position 0

        input_ids_padded.append(x["input_ids"] + [pad_token_id] * (max_seq - seq_len))
        attention_mask_padded.append(x["attention_mask"] + [0] * (max_seq - seq_len))

        # Pad word-level tensors to max_words
        word_starts_padded.append(x["word_starts"] + [0] * (max_words - n_words))
        heads_padded.append(x["heads"] + [0] * (max_words - n_words))
        rels_padded.append(x["rels"] + [0] * (max_words - n_words))
        # word_mask: 1 for real words only (NOT root, NOT padding) — used for loss/eval
        word_mask.append(x["is_real_word"] + [0] * (max_words - n_words))

    return {
        "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
        "word_starts": torch.tensor(word_starts_padded, dtype=torch.long),
        "heads": torch.tensor(heads_padded, dtype=torch.long),
        "rels": torch.tensor(rels_padded, dtype=torch.long),
        "word_mask": torch.tensor(word_mask, dtype=torch.bool),
    }
