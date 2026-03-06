"""
CoNLL-U format reader for Universal Dependencies treebanks.

CoNLL-U fields (tab-separated):
  ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC

This module reads .conllu files and returns a list of sentences.
Each sentence is a list of Token objects.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Token:
    id: int           # 1-based word index
    form: str         # surface form of the word
    upos: str         # Universal POS tag
    head: int         # index of syntactic head (0 = root)
    deprel: str       # dependency relation label
    is_multiword: bool = False   # True if this is a multi-word token line (e.g. "1-2")
    is_empty: bool = False       # True if this is an empty node (e.g. "1.1")


def read_conllu(path: str) -> List[List[Token]]:
    """
    Read a CoNLL-U file and return a list of sentences.
    Each sentence is a list of Token objects (excluding multi-word and empty nodes).

    Args:
        path: path to the .conllu file

    Returns:
        List of sentences, each sentence is a list of Token objects
    """
    sentences = []
    current = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#"):
                # comment line, skip
                continue

            if line == "":
                # blank line = sentence boundary
                if current:
                    sentences.append(current)
                    current = []
                continue

            fields = line.split("\t")
            if len(fields) < 8:
                continue

            raw_id = fields[0]

            # skip multi-word tokens like "1-2"
            if "-" in raw_id:
                continue

            # skip empty nodes like "1.1"
            if "." in raw_id:
                continue

            token = Token(
                id=int(raw_id),
                form=fields[1],
                upos=fields[3],
                head=int(fields[6]) if fields[6] != "_" else -1,
                deprel=fields[7],
            )
            current.append(token)

    # handle file that doesn't end with blank line
    if current:
        sentences.append(current)

    return sentences


def get_relation_vocab(sentences: List[List[Token]]) -> dict:
    """
    Build a mapping from dependency relation string to integer index.
    Index 0 is reserved for padding.

    Args:
        sentences: list of sentences from read_conllu()

    Returns:
        dict mapping relation string -> integer index (1-based)
    """
    relations = set()
    for sentence in sentences:
        for token in sentence:
            relations.add(token.deprel)

    # Sort for determinism, pad index = 0
    rel_list = sorted(relations)
    return {rel: idx + 1 for idx, rel in enumerate(rel_list)}


def print_stats(sentences: List[List[Token]], name: str = ""):
    """Print basic statistics about a dataset."""
    n_sent = len(sentences)
    n_tok = sum(len(s) for s in sentences)
    avg_len = n_tok / n_sent if n_sent > 0 else 0
    print(f"{name}: {n_sent} sentences, {n_tok} tokens, avg length {avg_len:.1f}")
