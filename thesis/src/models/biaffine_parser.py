"""
Biaffine Dependency Parser (Dozat & Manning, 2017).

Architecture:
    1. Encode tokens with BERT (+ optional BiLSTM for frozen variant)
    2. Apply separate MLP projections for arc and relation scoring
    3. Score all (head, dependent) pairs with biaffine attention
    4. At inference time, decode the best tree with Chu-Liu/Edmonds MST

Reference:
    Dozat, T. and Manning, C. D. (2017). Deep Biaffine Attention for
    Neural Dependency Parsing. ICLR 2017.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from models.encoder import BERTEncoder


class MLP(nn.Module):
    """Single-layer MLP with ELU activation and dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.33):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class BiaffineAttention(nn.Module):
    """
    Biaffine attention layer for scoring (head, dependent) pairs.

    For arc scoring (n_labels=1):
        score(i,j) = dep_i^T W head_j + bias^T head_j

    For relation scoring (n_labels=n_rels):
        score(i,j,r) = dep_i^T W_r head_j + bias_r^T head_j
    """

    def __init__(self, in_dim: int, n_labels: int = 1):
        super().__init__()
        self.n_labels = n_labels
        # Weight matrix: [n_labels, in_dim+1, in_dim]
        self.weight = nn.Parameter(torch.zeros(n_labels, in_dim + 1, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, dep: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dep:  [batch, n_words, in_dim]  (dependent representations)
            head: [batch, n_words, in_dim]  (head representations)

        Returns:
            scores: [batch, n_words, n_words] if n_labels==1
                    [batch, n_words, n_words, n_labels] if n_labels>1
        """
        batch, n_words, dim = dep.shape

        # Append a bias term to dep: [batch, n_words, in_dim+1]
        ones = dep.new_ones(*dep.shape[:-1], 1)
        dep_bias = torch.cat([dep, ones], dim=-1)

        # dep_bias: [batch, n_words, in_dim+1]
        # weight:   [n_labels, in_dim+1, in_dim]
        # For each label: score = dep_bias @ W @ head^T

        if self.n_labels == 1:
            # W: [in_dim+1, in_dim]
            W = self.weight.squeeze(0)  # [in_dim+1, in_dim]
            # dep_bias @ W: [batch, n_words, in_dim]
            inter = torch.matmul(dep_bias, W)
            # inter @ head^T: [batch, n_words, n_words]
            scores = torch.bmm(inter, head.transpose(1, 2))
        else:
            # W: [n_labels, in_dim+1, in_dim]
            # dep_bias: [batch, n_words, in_dim+1]
            # Reshape for batched matmul
            # inter: [batch, n_words, n_labels, in_dim]
            inter = torch.einsum("bni,lij->bnlj", dep_bias, self.weight)
            # head: [batch, n_words, in_dim] -> [batch, 1, n_words, in_dim]
            head_exp = head.unsqueeze(1)
            # scores: [batch, n_words_dep, n_labels, n_words_head]
            scores = torch.einsum("bnlj,bmj->bnlm", inter, head)
            # Rearrange to [batch, n_words_dep, n_words_head, n_labels]
            scores = scores.permute(0, 1, 3, 2)

        return scores


class BiaffineParser(nn.Module):
    """
    Full biaffine dependency parser.

    Args:
        encoder: BERTEncoder instance
        n_rels: number of dependency relation types (excluding padding)
        arc_dim: hidden dimension for arc MLP
        rel_dim: hidden dimension for relation MLP
        bilstm_layers: number of BiLSTM layers (used only when encoder is frozen)
        bilstm_hidden: hidden size of each BiLSTM direction
        mlp_dropout: dropout in MLP layers
        use_bilstm: if True, add BiLSTM between encoder and biaffine attention
    """

    def __init__(
        self,
        encoder: BERTEncoder,
        n_rels: int,
        arc_dim: int = 500,
        rel_dim: int = 100,
        bilstm_layers: int = 3,
        bilstm_hidden: int = 400,
        mlp_dropout: float = 0.33,
        use_bilstm: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_bilstm = use_bilstm

        # Input dimension to the biaffine scorer
        if use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=encoder.hidden_size,
                hidden_size=bilstm_hidden,
                num_layers=bilstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.33 if bilstm_layers > 1 else 0,
            )
            parser_input_dim = bilstm_hidden * 2
        else:
            parser_input_dim = encoder.hidden_size

        # Four MLPs: arc (head/dep) and relation (head/dep)
        self.mlp_arc_head = MLP(parser_input_dim, arc_dim, mlp_dropout)
        self.mlp_arc_dep  = MLP(parser_input_dim, arc_dim, mlp_dropout)
        self.mlp_rel_head = MLP(parser_input_dim, rel_dim, mlp_dropout)
        self.mlp_rel_dep  = MLP(parser_input_dim, rel_dim, mlp_dropout)

        # Biaffine attention layers
        self.arc_attn = BiaffineAttention(arc_dim, n_labels=1)
        self.rel_attn = BiaffineAttention(rel_dim, n_labels=n_rels)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Get word-level representations from encoder (+ optional BiLSTM)."""
        x = self.encoder(input_ids, attention_mask, word_starts)
        if self.use_bilstm:
            x, _ = self.bilstm(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        word_mask: torch.Tensor,
        heads: Optional[torch.Tensor] = None,
        rels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            word_starts: [batch, n_words]
            word_mask: [batch, n_words] boolean mask (True for real words)
            heads: [batch, n_words] gold heads (for loss computation)
            rels: [batch, n_words] gold relations (for loss computation)

        Returns:
            s_arc: [batch, n_words, n_words] arc scores
            s_rel: [batch, n_words, n_words, n_rels] relation scores
            loss: scalar training loss (None during inference)
        """
        x = self.encode(input_ids, attention_mask, word_starts)

        # Biaffine arc scoring
        arc_h = self.mlp_arc_head(x)
        arc_d = self.mlp_arc_dep(x)
        s_arc = self.arc_attn(arc_d, arc_h)  # [batch, n_words, n_words]

        # Biaffine relation scoring
        rel_h = self.mlp_rel_head(x)
        rel_d = self.mlp_rel_dep(x)
        s_rel = self.rel_attn(rel_d, rel_h)  # [batch, n_words, n_words, n_rels]

        loss = None
        if heads is not None and rels is not None:
            loss = self._compute_loss(s_arc, s_rel, heads, rels, word_mask)

        return s_arc, s_rel, loss

    def _compute_loss(
        self,
        s_arc: torch.Tensor,
        s_rel: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for arc and relation prediction.

        Arc loss: for each word, cross-entropy over all possible heads.
        Rel loss: for each word, cross-entropy over all relations at the gold head.
        """
        batch, n_words, _ = s_arc.shape

        # Valid head positions = root (position 0) + real words.
        # Padding positions should be masked to -inf in the head dimension.
        # word_mask is True only for real words (not root, not padding).
        # We must keep root (position 0) as a valid head target.
        valid_head = word_mask.clone()
        valid_head[:, 0] = True  # root is always a valid head
        head_mask = valid_head.unsqueeze(1).expand_as(s_arc)
        s_arc = s_arc.masked_fill(~head_mask, float("-inf"))

        # Arc loss: cross-entropy over head positions for each word
        # Flatten to [batch*n_words, n_words]
        arc_flat = s_arc.view(batch * n_words, n_words)
        head_flat = heads.view(batch * n_words)
        mask_flat = word_mask.view(batch * n_words)

        arc_loss = nn.functional.cross_entropy(
            arc_flat[mask_flat],
            head_flat[mask_flat],
            ignore_index=-1,
        )

        # Relation loss: pick scores at the gold head, then cross-entropy over rels
        # s_rel: [batch, n_words_dep, n_words_head, n_rels]
        # We need s_rel[b, i, gold_head[b,i], :] for each b,i
        head_idx = heads.unsqueeze(-1).unsqueeze(-1).expand(
            batch, n_words, 1, s_rel.size(-1)
        )
        # Gather at gold head: [batch, n_words, 1, n_rels] -> [batch, n_words, n_rels]
        rel_at_head = s_rel.gather(2, head_idx).squeeze(2)

        rel_flat = rel_at_head.view(batch * n_words, -1)
        rels_flat = rels.view(batch * n_words)

        rel_loss = nn.functional.cross_entropy(
            rel_flat[mask_flat],
            rels_flat[mask_flat],
            ignore_index=0,  # 0 is the padding relation index
        )

        return arc_loss + rel_loss

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> Tuple[List, List]:
        """
        Decode dependency trees for a batch of sentences.

        Returns:
            pred_heads: list of lists of predicted head indices (one per sentence)
            pred_rels: list of lists of predicted relation indices (one per sentence)
        """
        s_arc, s_rel, _ = self.forward(
            input_ids, attention_mask, word_starts, word_mask
        )

        batch, n_words, _ = s_arc.shape
        pred_heads = []
        pred_rels = []

        for b in range(batch):
            mask = word_mask[b]  # [n_words]: 1 for real words, 0 for root/padding
            n = mask.sum().item()  # number of real words

            # Tensor layout: position 0 = root, positions 1..n = real words.
            # Dependents: real words at positions 1..n  -> rows 1:n+1
            # Heads: root + real words at positions 0..n -> cols 0:n+1
            arc_scores = s_arc[b, 1:n+1, :n+1].cpu().numpy()  # [n, n+1]

            # Decode MST: returns head indices in 0..n (0=root)
            h = mst_decode(arc_scores)  # list of n head indices

            pred_heads.append(h)

            # Pick relation at predicted head for each real word
            rel_scores = s_rel[b, 1:n+1, :n+1, :]  # [n, n+1, n_rels]
            h_tensor = torch.tensor(h, device=s_rel.device)
            # Gather at predicted head: [n, n_rels]
            rel_at_pred = rel_scores[torch.arange(n), h_tensor]
            pred_r = rel_at_pred.argmax(-1).cpu().tolist()
            pred_rels.append(pred_r)

        return pred_heads, pred_rels


# ---------------------------------------------------------------------------
# Chu-Liu/Edmonds Maximum Spanning Tree decoder
# ---------------------------------------------------------------------------

def mst_decode(scores: np.ndarray) -> list:
    """
    Greedy maximum spanning tree decoder.

    Args:
        scores: [n_dep, n_head] float array where scores[i, j] = score of
                dependent word i having head j. n_head = n_dep + 1 because
                position 0 is the virtual root (root column included).

    Returns:
        heads: list of n_dep ints, each in 0..n_dep (0 = root).
    """
    scores = scores.copy()
    n_dep, n_head = scores.shape

    # Disallow self-loops: dependent i is at tensor position i+1, so head i+1
    for i in range(n_dep):
        if i + 1 < n_head:
            scores[i, i + 1] = float("-inf")

    # Greedy argmax: for each dependent, pick best head
    heads = scores.argmax(axis=1).tolist()  # heads[i] = argmax_j scores[i,j]
    return heads
