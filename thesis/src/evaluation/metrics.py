"""
Evaluation metrics for dependency parsing.

LAS (Labeled Attachment Score):
    Percentage of words where both head AND relation are predicted correctly.

UAS (Unlabeled Attachment Score):
    Percentage of words where the head is predicted correctly.

Following Universal Dependencies convention, punctuation tokens (UPOS == PUNCT)
are excluded from evaluation.
"""

from typing import List, Tuple
from data.conllu_reader import Token


class ParseEvaluator:
    """
    Accumulates predictions across batches and computes LAS/UAS.

    Usage:
        evaluator = ParseEvaluator()
        for batch in dataloader:
            pred_heads, pred_rels = model.predict(...)
            evaluator.update(sentences_in_batch, pred_heads, pred_rels, rel_vocab)
        las, uas = evaluator.get_scores()
        evaluator.reset()
    """

    def __init__(self):
        self.correct_las = 0
        self.correct_uas = 0
        self.total = 0

    def reset(self):
        self.correct_las = 0
        self.correct_uas = 0
        self.total = 0

    def update(
        self,
        sentences: List[List[Token]],
        pred_heads: List[List[int]],
        pred_rels: List[List[int]],
        rel_vocab: dict,
    ):
        """
        Add a batch of predictions to the running counts.

        Args:
            sentences: list of sentences (gold annotations)
            pred_heads: list of lists of predicted head indices (0-based)
            pred_rels: list of lists of predicted relation indices
            rel_vocab: dict mapping relation string -> integer index
        """
        # Build reverse mapping: index -> relation string
        idx_to_rel = {v: k for k, v in rel_vocab.items()}

        for sent, p_heads, p_rels in zip(sentences, pred_heads, pred_rels):
            for i, token in enumerate(sent):
                # Skip punctuation following UD convention
                if token.upos == "PUNCT":
                    continue

                if i >= len(p_heads):
                    # Sentence was truncated during tokenization; skip remaining
                    break

                self.total += 1
                gold_head = token.head          # 0-based (0=root)
                gold_rel = token.deprel

                pred_h = p_heads[i]
                pred_r_idx = p_rels[i]
                pred_rel = idx_to_rel.get(pred_r_idx, "")

                if pred_h == gold_head:
                    self.correct_uas += 1
                    if pred_rel == gold_rel:
                        self.correct_las += 1

    def get_scores(self) -> Tuple[float, float]:
        """
        Returns:
            (LAS, UAS) as percentages (0-100).
        """
        if self.total == 0:
            return 0.0, 0.0
        las = 100.0 * self.correct_las / self.total
        uas = 100.0 * self.correct_uas / self.total
        return las, uas

    def __repr__(self):
        las, uas = self.get_scores()
        return f"LAS={las:.2f}  UAS={uas:.2f}  (over {self.total} tokens)"


def evaluate_file(pred_path: str, gold_path: str) -> Tuple[float, float]:
    """
    Evaluate a CoNLL-U prediction file against a gold file.
    Both files must have the same sentences in the same order.

    This is an alternative to the online evaluator — useful when you
    write predictions to disk and want to evaluate them later.

    Args:
        pred_path: path to predicted .conllu file
        gold_path: path to gold .conllu file

    Returns:
        (LAS, UAS) as percentages
    """
    from data.conllu_reader import read_conllu

    gold_sents = read_conllu(gold_path)
    pred_sents = read_conllu(pred_path)

    evaluator = ParseEvaluator()
    # Build a dummy rel_vocab that maps relation string -> itself for direct comparison
    dummy_vocab = {rel: rel for s in gold_sents for t in s for rel in [t.deprel]}

    # Rebuild pred as heads/rels integers matching this dummy vocab
    for g_sent, p_sent in zip(gold_sents, pred_sents):
        p_heads = [t.head for t in p_sent]
        p_rels = [dummy_vocab.get(t.deprel, "") for t in p_sent]
        evaluator.update([g_sent], [p_heads], [p_rels], dummy_vocab)

    return evaluator.get_scores()
