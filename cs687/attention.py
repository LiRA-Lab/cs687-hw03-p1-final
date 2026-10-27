"""
Attention: the operation that lets a position read the positions before it.

This module is the attention of the first part of the Lecture 2 notes. The
notes build it in steps: a weighted average of the earlier vectors, weights
that come from the dot product of a query with the keys, a division by the
square root of the key width, a softmax, and a causal mask that hides the
future. This repository ships it complete, with the course's answers to the
three pieces that were Tasks 1 and 2 of Homework 2 inserted.
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    dropout: nn.Dropout | None = None,
) -> torch.Tensor:
    """Combine the values with weights computed from the queries and the keys.

    In the notation of the notes, the scores are E = Q K^T / sqrt(d_k), the
    causal mask M adds minus infinity above the diagonal, the weights are
    A = softmax(E + M) row by row, and the result is H = A V.

    Args:
        q: queries, shape (..., T, d_k). The leading axes may be a batch, or a
            batch and heads. The function must not care how many there are.
        k: keys, shape (..., T, d_k).
        v: values, shape (..., T, d_v).
        causal: whether to hide the future with the mask M.
        dropout: an optional dropout module applied to the weights A after the
            softmax and before the multiplication by v. Training uses it; in
            evaluation mode it changes nothing.

    Returns:
        The contextual vectors, shape (..., T, d_v).
    """
    # Supplied complete. This function was Task 1 of Homework 2.
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if causal:
        T = scores.shape[-1]
        allowed = torch.tril(torch.ones(T, T, device=q.device)).bool()
        scores = scores.masked_fill(~allowed, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v


class CausalSelfAttention(nn.Module):
    """One head of causal self-attention. Supplied complete.

    It projects the input vectors to queries, keys and values with three
    matrices and no biases, and calls the attention function above.

    Input:  (B, T, d_model)
    Output: (B, T, d_head)
    """

    def __init__(self, d_model: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        # The projections have no bias. A bias on the key adds the same
        # constant to every score in a row, which the softmax ignores.
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)                       # (B, T, d_head)
        k = self.key(x)                         # (B, T, d_head)
        v = self.value(x)                       # (B, T, d_head)
        return scaled_dot_product_attention(q, k, v, causal=True, dropout=self.drop)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Input:  (B, T, d_model)
    Output: (B, T, d_model)

    The heads partition the width. With n_heads heads and width d_model, each
    head works in a space of width d_head = d_model / n_heads, so the number
    of parameters does not depend on the number of heads. The query, key and
    value projections are one matrix each for all heads at once, the heads are
    separated by reshaping, and the output matrix W_o mixes them again.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 causal: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.causal = causal
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One matrix per role, covering all heads at once, and no biases.
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        # The output matrix W_o of the notes, which mixes the head outputs.
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the width into heads and move the head axis forward.

        Input:  (B, T, d_model)
        Output: (B, n_heads, T, d_head)
        """
        # Supplied complete. This method was Task 2 of Homework 2.
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Undo split_heads: place the heads side by side into one width axis.

        Input:  (B, n_heads, T, d_head)
        Output: (B, T, d_model)
        """
        # Supplied complete. This method was Task 2 of Homework 2. The
        # transpose leaves the tensor non-contiguous, so it is copied before
        # the view joins the last two axes.
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Supplied complete.
        q = self.split_heads(self.query(x))     # (B, n_heads, T, d_head)
        k = self.split_heads(self.key(x))       # (B, n_heads, T, d_head)
        v = self.split_heads(self.value(x))     # (B, n_heads, T, d_head)

        out = scaled_dot_product_attention(
            q, k, v, causal=self.causal, dropout=self.attn_drop
        )                                       # (B, n_heads, T, d_head)

        out = self.merge_heads(out)             # (B, T, d_model)
        return self.resid_drop(self.proj(out))
