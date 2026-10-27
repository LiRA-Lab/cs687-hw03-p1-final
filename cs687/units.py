"""
Unit conversions for reporting a language model's loss.

The same predictive performance can be summarized as loss in nats per token,
perplexity, bits per token, or bits per byte. The first three depend on the
tokenizer. Bits per byte compares models across tokenizers when both score
the same bytes. These functions are
complete. The notebook uses them in its units section, and the report asks
for the same conversion.
"""

import math


def perplexity(loss_nats: float) -> float:
    """Supplied complete. The number of equally likely choices that would
    give this mean loss. A uniform model over V tokens has perplexity V."""
    return math.exp(loss_nats)


def bits_per_token(loss_nats: float) -> float:
    """Supplied complete. Convert nats to bits, since log2(x) = ln(x) / ln(2)."""
    return loss_nats / math.log(2)


def bits_per_byte(loss_nats: float, tokens_per_byte: float) -> float:
    """Supplied complete. The unit for comparing models that use different
    tokenizers, when both score the same bytes of the same text."""
    return bits_per_token(loss_nats) * tokens_per_byte


def compression_ratio(loss_nats: float, tokens_per_byte: float) -> float:
    """Supplied complete. An ideal estimate of the fraction of the raw size that
    the text takes when written with the model's probabilities, with the model
    and tokenizer available to the decoder. Raw text uses 8 bits per byte."""
    return bits_per_byte(loss_nats, tokens_per_byte) / 8.0
