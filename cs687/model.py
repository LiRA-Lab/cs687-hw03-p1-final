"""
The whole decoder-only model, from token ids to logits.

This repository ships it complete, with the course's answer to the forward
pass, which was Task 4 of Homework 2, inserted. Training the model and
decoding from it are the subject of this homework, in cs687/training.py and
cs687/sampling.py.
"""

import torch
import torch.nn as nn

from cs687.block import TransformerBlock
from cs687.embedding import InputEmbedding


class LanguageModel(nn.Module):
    """A decoder-only transformer.

    Input:  (B, T) token ids
    Output: (B, T, vocab_size) logits

    The input layer is the one Homework 1 built, a token lookup plus a
    position vector. Then n_layers blocks are applied in order, a final
    normalization standardizes every row, and the head of Lecture 1 turns each
    row into a score per vocabulary id. Row j of the logits scores the token
    at slot j + 1.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "max_len": max_len,
        }
        self.embed = InputEmbedding(vocab_size, d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm_final = nn.LayerNorm(d_model)      # LN_f of the notes
        self.head = nn.Linear(d_model, vocab_size, bias=False)   # W_u of the notes

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed the ids, apply every block in order, normalize once more,
        and project every row to the vocabulary. Shapes: (B, T) ->
        (B, T, d_model) -> ... -> (B, T, vocab_size)."""
        # Supplied complete. This method was Task 4 of Homework 2.
        x = self.embed(token_ids)                 # (B, T, d_model)
        for block in self.blocks:
            x = block(x)                          # (B, T, d_model)
        x = self.norm_final(x)
        return self.head(x)                       # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample tokens one at a time, appending each to the input. Supplied complete.

        The last row of the logits scores the token that follows the window,
        and that row is turned into probabilities and sampled from. This is
        the plain sampler of Homework 2. The decoding rules of this homework
        live in cs687/sampling.py, whose generate function replaces this one.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Never feed the model more positions than it has position vectors for.
            window = token_ids[:, -self.config["max_len"]:]
            logits = self(window)[:, -1, :]             # (B, vocab_size)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


def count_parameters(model: nn.Module) -> dict:
    """Break a model's parameter count down by component. Supplied complete."""
    groups = {"embeddings": 0, "attention": 0, "feed_forward": 0,
              "norms": 0, "head": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        if "embed" in name:
            groups["embeddings"] += n
        elif "attn" in name:
            groups["attention"] += n
        elif "ffn" in name:
            groups["feed_forward"] += n
        elif "norm" in name:
            groups["norms"] += n
        elif "head" in name:
            groups["head"] += n
    groups["total"] = sum(p.numel() for p in model.parameters())
    return groups
