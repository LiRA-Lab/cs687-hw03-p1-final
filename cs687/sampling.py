"""
Decoding: turning the model's distribution into text.

This module is the decoding section of the second part of the Lecture 2
notes. A trained model gives a distribution over the next id at every slot,
and what to do with it is a decision of its own, made after training. Three
rules act on the distribution before the draw, and the three are left for
you to write, in the order the notes teach them: temperature, which changes
the probabilities, and top-p and top-k, which remove ids. The notebook
attaches your versions to this module in place of the stubs, and the tests
check them against the distribution the notes print. The sampler that
combines the rules and the loop that produces text are supplied complete.
"""

from __future__ import annotations

import torch


# ----------------------------------------------------------------------
# Task 2: temperature
# ----------------------------------------------------------------------
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide the logits by the temperature tau.

    The notes' temperature: a temperature below one widens the gaps between
    the logits and sharpens the distribution toward its favourite, a
    temperature above one narrows them and flattens it toward uniform, and
    temperature one changes nothing.

    Args:
        logits: shape (..., V), the scores at one slot.
        temperature: a positive number. Zero and negative values are not
            meaningful here, since greedy decoding is handled separately by
            the sampler below, so raise a ValueError for them.

    Returns:
        A tensor of the same shape.
    """
    raise NotImplementedError(
        "Task 2 is written in the homework notebook, which attaches it to this module."
    )


# ----------------------------------------------------------------------
# Task 3: top-p
# ----------------------------------------------------------------------
def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of the most probable ids whose mass reaches p.

    The notes' top-p: take the ids in order of probability from the most
    probable down, keep them until the mass they carry together reaches p,
    and remove every other id. The removal is made on the logits, by setting
    the logit of every removed id to minus infinity, which is the device of
    the causal mask: the softmax then gives those ids exactly zero and
    renormalizes the survivors on its own.

    One detail decides the edge case the notes name. The condition that
    removes an id is that the mass before it already reaches p, that is, is
    at least p, and not that the mass including it does, so that the
    favourite is kept even when it alone carries more than p. Written the
    other way round, the rule keeps nothing. The ids are taken in the order
    the sort gives them, so when ids of equal probability meet at the cut,
    the sort decides which of them is kept, and the set holds exactly as
    many ids as the running mass gives.

    Args:
        logits: shape (..., V).
        p: the cumulative probability threshold that the kept set must
            reach, between 0 and 1. If p is at least 1, return the logits
            unchanged.

    Returns:
        A tensor of the same shape with the removed ids at minus infinity
        and every kept id's logit unchanged.

    Library functions you may not know: torch.sort with descending=True
    returns the sorted values and the index each one came from; cumsum along
    the last axis gives running sums; a boolean tensor's scatter along the
    last axis with those indices puts a mask over the sorted order back into
    the original order; masked_fill replaces the entries where a mask is
    true.
    """
    raise NotImplementedError(
        "Task 3 is written in the homework notebook, which attaches it to this module."
    )


# ----------------------------------------------------------------------
# Task 4: top-k
# ----------------------------------------------------------------------
def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k most probable ids and remove the rest.

    The notes' top-k: the k ids with the largest logits are kept, whatever
    mass they carry, and every other id is removed by setting its logit to
    minus infinity, as in top-p. An id whose logit equals the k-th largest
    is kept too, so when ids tie at the k-th place, more than k survive.

    Args:
        logits: shape (..., V).
        k: how many ids to keep. If k is zero or at least the vocabulary
            size, return the logits unchanged.

    Returns:
        A tensor of the same shape with the removed ids at minus infinity
        and every kept id's logit unchanged.

    Library functions you may not know: torch.topk returns the k largest
    values of each row and their indices, in descending order, so the
    smallest kept value of a row is the last of them; masked_fill replaces
    the entries where a mask is true.
    """
    raise NotImplementedError(
        "Task 4 is written in the homework notebook, which attaches it to this module."
    )


# ----------------------------------------------------------------------
# Supplied: the sampler and the loop
# ----------------------------------------------------------------------
@torch.no_grad()
def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """Choose the next id from the logits of the last slot. Supplied complete.

    Greedy decoding takes the favourite. Otherwise the rules are applied in
    the order the notes give: temperature first, then the cut, top-k and then
    top-p when both are asked for, and the draw is made from what is left.

    Args:
        logits: shape (B, V).

    Returns:
        Ids of shape (B, 1).
    """
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """Produce text one id at a time. Supplied complete.

    The loop of the notes: run the prefix through the model, read the
    distribution at the last slot, pick an id by the rule, append it, and run
    the model again on the longer prefix. The model never sees more than its
    window length T, so the prefix is cut to its last T ids before each pass.

    Args:
        prompt_ids: shape (B, T0), the ids of the prompt.

    Returns:
        The prompt followed by max_new_tokens new ids, shape (B, T0 + max_new_tokens).
    """
    model.eval()
    ids = prompt_ids
    window = model.config["max_len"]
    for _ in range(max_new_tokens):
        logits = model(ids[:, -window:])[:, -1, :]
        next_id = sample_next(logits, temperature, top_k, top_p, greedy)
        ids = torch.cat([ids, next_id], dim=1)
    return ids
