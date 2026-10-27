"""
The transformer block: attention, the feed-forward layer, and the two devices
that hold a deep stack together, residual connections and layer normalization.

This repository ships it complete, with the course's answer to the block's
forward pass, which was Task 3 of Homework 2, inserted.
"""

import torch
import torch.nn as nn

from cs687.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """The feed-forward layer of the notes. Supplied complete.

    It reads one vector and writes one vector, with the same parameters at
    every position, and it never looks at a neighbouring slot. The hidden
    width d_ff is four times d_model, which is where most of a block's
    parameters live.

    Input and output are both (B, T, d_model).
    """

    def __init__(self, d_model: int, multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, multiplier * d_model),     # W_1 and b_1
            nn.GELU(),                                    # the nonlinearity g
            nn.Linear(multiplier * d_model, d_model),     # W_2 and b_2
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block in the pre-norm arrangement.

    Each of the two operations reads a normalized copy of the residual stream
    and adds its result back to the stream. The stream itself is never
    overwritten, which is what lets a deep stack train.

    Input and output are both (B, T, d_model).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 causal: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal=causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two steps of the block, X' = X + Attn(LN_1(X)) and
        X'' = X' + f(LN_2(X')), and return X''."""
        # Supplied complete. This method was Task 3 of Homework 2.
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
