"""CS 687 Homework 3: training the model and decoding from it, on the complete Homework 1 and 2 code."""

# Check the documented minimum before importing the course modules.
import sys as _sys

if _sys.version_info < (3, 10):
    raise RuntimeError(
        "This repository requires Python 3.10 or later. "
        "The current interpreter is "
        + ".".join(str(_part) for _part in _sys.version_info[:3])
        + f" at {_sys.executable}."
    )

from cs687.tokenizer import BPETokenizer
from cs687.data import NextTokenDataset, make_loader
from cs687.embedding import InputEmbedding
from cs687.units import perplexity, bits_per_token, bits_per_byte
from cs687.attention import (
    CausalSelfAttention,
    MultiHeadAttention,
    scaled_dot_product_attention,
)
from cs687.block import FeedForward, TransformerBlock
from cs687.model import LanguageModel, count_parameters
from cs687.training import (
    train_step,
    learning_rate_at,
    save_checkpoint,
    load_checkpoint,
    sample_batch,
    held_out_loss,
    train_run,
)
from cs687.sampling import (
    apply_temperature,
    top_p_filter,
    top_k_filter,
    sample_next,
    generate,
)

__all__ = [
    "BPETokenizer",
    "NextTokenDataset",
    "make_loader",
    "InputEmbedding",
    "perplexity",
    "bits_per_token",
    "bits_per_byte",
    "scaled_dot_product_attention",
    "CausalSelfAttention",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "LanguageModel",
    "count_parameters",
    "train_step",
    "learning_rate_at",
    "save_checkpoint",
    "load_checkpoint",
    "sample_batch",
    "held_out_loss",
    "train_run",
    "apply_temperature",
    "top_p_filter",
    "top_k_filter",
    "sample_next",
    "generate",
]
