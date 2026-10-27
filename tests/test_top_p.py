"""Tests for Task 3: top_p_filter.

The notebook runs these for you. They use the distribution of eight ids that
the notes print, and one of them needs Task 2. Each test reaches the function
through the module, so that the version the notebook attached is the one
tested.
"""

import pytest
import torch

from cs687 import sampling

# The logits of the notes' table, over eight ids.
LOGITS = torch.tensor([[3.2, 2.1, 1.8, 0.9, 0.4, 0.1, -0.6, -1.2]])
TOKENS = ["the", "cat", "dog", "house", "idea", "running", "purple", "seven"]


def cut(logits, p):
    return sampling.top_p_filter(logits.clone(), p)


def kept(logits):
    return [TOKENS[i] for i in range(len(TOKENS)) if torch.isfinite(logits[0, i])]


def probabilities(logits):
    return torch.softmax(logits, dim=-1)[0]


def test_at_p_09_four_ids_are_kept():
    """The cumulative mass runs 0.549, 0.732, 0.868, 0.923: the fourth id carries it past 0.9."""
    survivors = kept(cut(LOGITS, 0.9))
    assert survivors == ["the", "cat", "dog", "house"], (
        f"At p = 0.9 the kept ids should be the, cat, dog and house, but they are {survivors}. "
        "The ids are taken from the most probable down until their mass reaches p, and the id "
        "that carries the sum past p is kept."
    )


def test_at_p_075_three_ids_are_kept():
    """At p = 0.75 the mass is reached after three ids: the set is decided by the distribution."""
    survivors = kept(cut(LOGITS, 0.75))
    assert survivors == ["the", "cat", "dog"], (
        f"At p = 0.75 the kept ids should be the, cat and dog, but they are {survivors}."
    )


def test_at_p_05_the_favourite_alone_is_kept():
    """The favourite carries 0.55 on its own, which already reaches p = 0.5."""
    survivors = kept(cut(LOGITS, 0.5))
    assert survivors == ["the"], (
        f"At p = 0.5 only the favourite should be kept, but the kept ids are {survivors}."
    )


def test_the_survivors_are_renormalized_by_the_softmax():
    """After the cut at p = 0.9 the favourite has 0.595 and the removed ids have exactly zero."""
    out = probabilities(cut(LOGITS, 0.9))
    assert out.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert out[0].item() == pytest.approx(0.595, abs=1e-3), (
        f"After the cut the favourite should have probability 0.595, not {out[0].item():.3f}. "
        "The removed ids are set to minus infinity, so the softmax renormalizes the survivors."
    )
    assert out[4:].sum().item() == pytest.approx(0.0, abs=1e-7), (
        "The removed ids must have exactly zero probability after the softmax, which only "
        "minus infinity gives."
    )


def test_the_kept_logits_are_unchanged():
    """The cut removes ids and changes nothing else: the four kept ids return with their own logits."""
    out = cut(LOGITS, 0.9)
    assert torch.allclose(out[0, :4], LOGITS[0, :4]), (
        f"The kept ids should keep their logits, {[round(v, 3) for v in LOGITS[0, :4].tolist()]}, "
        f"but the function returned {[round(v, 3) for v in out[0, :4].tolist()]}. The function "
        "returns logits, not probabilities or log-probabilities: only the removed ids change, to "
        "minus infinity."
    )


def test_the_favourite_is_kept_when_it_alone_exceeds_p():
    """The edge case the notes name: a cut that removes everything leaves nothing to draw from."""
    peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    assert probabilities(peaked)[0].item() > 0.99
    out = cut(peaked, 0.5)
    assert torch.isfinite(out).sum().item() >= 1, (
        "Every id was removed. The condition that removes an id is that the mass before it "
        "already reaches p, not that the mass including it does."
    )
    assert torch.isfinite(out[0, 0]), "The most probable id must always survive the cut."


def test_a_mass_that_exactly_reaches_p_closes_the_set():
    """Four equal ids: the first two carry exactly 0.5, which reaches p = 0.5, so two ids are kept."""
    count = torch.isfinite(cut(torch.zeros(1, 4), 0.5)).sum().item()
    assert count == 2, (
        f"With four ids of probability 0.25 each and p = 0.5, {count} ids were kept instead of 2. "
        "The first two carry exactly 0.5, which reaches p, so the third is removed: an id is "
        "removed when the mass before it is at least p. The sort decides which two of the equal "
        "ids come first, and only their number is checked."
    )


def test_ids_tied_at_the_cut_are_taken_in_the_sort_order():
    """Probabilities 0.4, 0.3 and 0.3 at p = 0.6: the favourite and one of the two tied ids are kept."""
    tied = torch.tensor([[4.0, 3.0, 3.0]]).log()
    out = cut(tied, 0.6)
    count = torch.isfinite(out).sum().item()
    assert torch.isfinite(out[0, 0]) and count == 2, (
        f"At p = 0.6 the favourite and one of the two tied ids should be kept, two ids in all, "
        f"but {count} were kept. The mass before the second tied id is 0.7, which reaches p, so "
        "it is removed although its probability equals that of the id kept before it."
    )


def test_a_sharper_distribution_keeps_fewer_ids():
    """Temperature 0.5 first, then the same p = 0.9, keeps two ids instead of four."""
    wide = torch.isfinite(cut(LOGITS, 0.9)).sum().item()
    sharp = torch.isfinite(cut(sampling.apply_temperature(LOGITS, 0.5), 0.9)).sum().item()
    assert wide == 4 and sharp == 2, (
        f"At p = 0.9 the plain distribution should keep 4 ids and the sharpened one 2, but "
        f"they keep {wide} and {sharp}. The size of the kept set follows the distribution."
    )


def test_the_order_of_the_ids_does_not_matter():
    """The same eight logits in a shuffled order keep the same four ids at p = 0.9."""
    order = [5, 0, 7, 2, 4, 1, 6, 3]
    shuffled = LOGITS[:, order]
    out = cut(shuffled, 0.9)
    survivors = sorted(TOKENS[order[i]] for i in range(len(order)) if torch.isfinite(out[0, i]))
    assert survivors == sorted(["the", "cat", "dog", "house"]), (
        f"With the ids in a different order the kept ids should still be the, cat, dog and house, "
        f"but they are {survivors}. The cut is decided in sorted order and must be carried back "
        "to the original order of the ids."
    )


def test_p_of_one_changes_nothing():
    assert torch.allclose(cut(LOGITS, 1.0), LOGITS), (
        "With p at least one every id is kept and the logits come back unchanged."
    )
