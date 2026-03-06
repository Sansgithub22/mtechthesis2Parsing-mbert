"""
BERT/XLM-R encoder with scalar mixing of layers.

The paper uses a weighted sum of all 12 BERT layers as the CWR for each token.
This is the "scalar mix" technique from Peters et al. (2018).

For the frozen (non-FT) variant:
    BERT weights are frozen; a BiLSTM processes the scalar-mixed output.

For the fine-tuned (FT) variant:
    BERT weights are updated during parser training; no BiLSTM is used.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class ScalarMix(nn.Module):
    """
    Compute a weighted sum of BERT layer outputs.
    Weights are learned via softmax over scalar parameters.

    Output = gamma * sum_k( softmax(s)[k] * layer_k )
    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.scalars = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: tuple) -> torch.Tensor:
        """
        Args:
            hidden_states: tuple of [batch, seq_len, hidden_size] tensors,
                           one per layer (including embedding layer)

        Returns:
            [batch, seq_len, hidden_size] weighted sum
        """
        # Stack: [n_layers, batch, seq_len, hidden_size]
        stacked = torch.stack(hidden_states, dim=0)
        weights = torch.softmax(self.scalars, dim=0)  # [n_layers]

        # Weighted sum over layers: broadcast weights over batch/seq dims
        mixed = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        return self.gamma * mixed


class BERTEncoder(nn.Module):
    """
    BERT or XLM-R encoder that returns word-level representations.

    For each word, we take the first wordpiece's representation (a standard
    approach for parsing, following Kondratyuk & Straka, 2019).

    Args:
        model_name: HuggingFace model identifier
            - mBERT: 'bert-base-multilingual-cased'
            - XLM-R: 'xlm-roberta-base'
        freeze: if True, freeze all BERT parameters (non-FT variant)
        dropout: dropout rate applied to encoder output
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        freeze: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        n_layers = self.bert.config.num_hidden_layers + 1  # +1 for embedding layer
        self.scalar_mix = ScalarMix(n_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_starts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of sentences and return word-level representations.

        Args:
            input_ids: [batch, seq_len] wordpiece token IDs
            attention_mask: [batch, seq_len] 1 for real tokens, 0 for padding
            word_starts: [batch, n_words] index of first wordpiece for each word

        Returns:
            [batch, n_words, hidden_size] word-level representations
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # hidden_states: tuple of (n_layers+1) tensors, each [batch, seq_len, hidden]
        hidden_states = outputs.hidden_states

        # Scalar mix over all layers
        mixed = self.scalar_mix(hidden_states)  # [batch, seq_len, hidden]
        mixed = self.dropout(mixed)

        # Gather first-wordpiece representation for each word
        # word_starts: [batch, n_words] — index into seq_len dimension
        batch_size, n_words = word_starts.shape
        hidden_size = mixed.size(-1)

        # Expand word_starts to gather from hidden dim
        indices = word_starts.unsqueeze(-1).expand(batch_size, n_words, hidden_size)
        word_repr = mixed.gather(dim=1, index=indices)  # [batch, n_words, hidden]

        return word_repr
