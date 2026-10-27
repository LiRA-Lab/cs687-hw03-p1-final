"""Tests for Task 4: top_k_filter.

The notebook runs these for you. They use the distribution of eight ids that
the notes print. Each test reaches the function through the module, so that
the version the notebook attached is the one tested. The last test draws
from the supplied sampler, which calls your filter.
"""

import pytest
import torch

from cs687 import sampling

# The logits of the notes' table, over eight ids.
LOGITS = torch.tensor([[3.2, 2.1, 1.8, 0.9, 0.4, 0.1, -0.6, -1.2]])
TOKENS = ["the", "cat", "dog", "house", "idea", "running", "purple", "seven"]


def cut(logits, k):
    return sampling.top_k_filter(logits.clone(), k)


def kept(logits):
    return [TOKENS[i] for i in range(len(TOKENS)) if torch.isfinite(logits[0, i])]


def probabilities(logits):
    return torch.softmax(logits, dim=-1)[0]


def test_k_ids_are_kept_when_none_tie():
    """The table has no ties, so k = 3 keeps exactly three ids."""
    count = torch.isfinite(cut(LOGITS, 3)).sum().item()
    assert count == 3, f"With k = 3 exactly three ids should keep a finite logit, but {count} do."


def test_the_k_most_probable_ids_are_the_ones_kept():
    """With k = 3 the survivors are the, cat and dog."""
    survivors = kept(cut(LOGITS, 3))
    assert survivors == ["the", "cat", "dog"], (
        f"With k = 3 the kept ids should be the, cat and dog, but they are {survivors}."
    )


def test_every_id_tied_with_the_kth_is_kept():
    """Logits 3, 2, 2 and 1 at k = 2: the two ids tied at the second place are both kept."""
    tied = torch.tensor([[3.0, 2.0, 2.0, 1.0]])
    survivors = torch.isfinite(cut(tied, 2))[0].tolist()
    assert survivors == [True, True, True, False], (
        f"With k = 2 the ids kept should be the first three, because two ids tie at the second "
        f"place, but the kept ones are {survivors}. Every id whose logit equals the k-th largest "
        "is kept."
    )


def test_the_survivors_are_renormalized_by_the_softmax():
    """The notes' numbers: the three survivors become 0.633, 0.211 and 0.156."""
    out = probabilities(cut(LOGITS, 3))
    assert out.sum().item() == pytest.approx(1.0, abs=1e-5)
    for index, expected in enumerate([0.633, 0.211, 0.156]):
        assert out[index].item() == pytest.approx(expected, abs=1e-3), (
            f"After the cut '{TOKENS[index]}' should have probability {expected}, not "
            f"{out[index].item():.3f}. The removed ids are set to minus infinity, so the "
            "softmax renormalizes the survivors."
        )
    assert out[3:].sum().item() == pytest.approx(0.0, abs=1e-7), (
        "The removed ids must have exactly zero probability after the softmax."
    )


def test_the_kept_logits_are_unchanged():
    """The cut removes ids and changes nothing else: the three kept ids return with their own logits."""
    out = cut(LOGITS, 3)
    assert torch.allclose(out[0, :3], LOGITS[0, :3]), (
        f"The kept ids should keep their logits, {[round(v, 3) for v in LOGITS[0, :3].tolist()]}, "
        f"but the function returned {[round(v, 3) for v in out[0, :3].tolist()]}. The function "
        "returns logits, not probabilities or log-probabilities: only the removed ids change, to "
        "minus infinity."
    )


def test_k_of_zero_or_the_vocabulary_size_changes_nothing():
    for k in (0, 8, 50):
        assert torch.allclose(cut(LOGITS, k), LOGITS), (
            f"With k = {k} every id is kept and the logits come back unchanged."
        )


def test_k_of_one_is_greedy_decoding():
    out = probabilities(cut(LOGITS, 1))
    assert out[0].item() == pytest.approx(1.0, abs=1e-6), (
        "With k = 1 the favourite alone is kept, and the softmax gives it probability one."
    )


def test_the_sampler_never_draws_a_removed_id():
    """Two hundred draws under k = 2, through the supplied sampler, never leave the top two."""
    torch.manual_seed(0)
    seen = {sampling.sample_next(LOGITS, temperature=1.0, top_k=2).item() for _ in range(200)}
    assert seen <= {0, 1}, (
        f"The sampler drew the ids {sorted(seen)}, but under k = 2 only ids 0 and 1 may be drawn."
    )
