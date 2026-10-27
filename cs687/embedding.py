"""
The input layer: token embeddings plus learned absolute position embeddings.

This module is complete. It is given to you so that you can see how the token
lookup the notes describe appears in PyTorch, and so that you can check the
output shape in the homework.
"""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Token embedding plus learned absolute position embedding.

    Supplied complete.

    Input:  an integer tensor of shape (B, T) holding token ids.
    Output: a float tensor of shape (B, T, d_model).

    The notes write the token table as a matrix with one column per id, so
    that multiplying it by a one-hot vector selects the token's column.
    PyTorch stores the transpose, with one row per token, and nn.Embedding
    selects that row directly instead of performing the multiplication. The
    position table is a second lookup of the same kind, indexed by slot, and
    the two vectors are added. Dropout acts only in training mode.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.max_len}: the "
                "position table has no vector for a slot at or beyond max_len."
            )
        positions = torch.arange(seq_len, device=token_ids.device)
        input_vectors = self.tok(token_ids) + self.pos(positions)
        return self.drop(input_vectors)
