"""Tests for Task 2: apply_temperature.

The notebook runs these for you. They use the distribution of eight ids that
the notes print, so the numbers are the ones computed on the page. Each test
reaches the function through the module, so that the version the notebook
attached is the one tested.
"""

import pytest
import torch

from cs687 import sampling

# The logits of the notes' table, over eight ids.
LOGITS = torch.tensor([[3.2, 2.1, 1.8, 0.9, 0.4, 0.1, -0.6, -1.2]])
TOKENS = ["the", "cat", "dog", "house", "idea", "running", "purple", "seven"]


def temper(logits, temperature):
    return sampling.apply_temperature(logits, temperature)


def probabilities(logits):
    return torch.softmax(logits, dim=-1)[0]


def test_temperature_one_changes_nothing():
    out = temper(LOGITS, 1.0)
    assert torch.allclose(out, LOGITS), (
        "At temperature one the logits must come back unchanged, since dividing by one does nothing."
    )


def test_the_numbers_of_the_notes():
    """At tau = 0.5 the favourite takes 0.84, and at tau = 1.5 it falls to 0.40."""
    sharp = probabilities(temper(LOGITS, 0.5))
    assert sharp[0].item() == pytest.approx(0.842, abs=1e-3), (
        f"At temperature 0.5 the favourite should have probability 0.842, not {sharp[0].item():.3f}."
    )
    assert sharp[2].item() == pytest.approx(0.051, abs=1e-3), (
        f"At temperature 0.5 'dog' should have probability 0.051, not {sharp[2].item():.3f}."
    )
    flat = probabilities(temper(LOGITS, 1.5))
    assert flat[0].item() == pytest.approx(0.399, abs=1e-3), (
        f"At temperature 1.5 the favourite should have probability 0.399, not {flat[0].item():.3f}."
    )


def test_a_low_temperature_approaches_greedy_decoding():
    out = probabilities(temper(LOGITS, 0.01))
    assert out[0].item() > 0.999, (
        "As the temperature goes to zero the favourite should take all of the mass."
    )


def test_a_high_temperature_approaches_a_uniform_draw():
    out = probabilities(temper(LOGITS, 100.0))
    assert out.max().item() - out.min().item() < 0.02, (
        "As the temperature grows every id should become about equally likely."
    )


def test_zero_and_negative_temperatures_are_rejected():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            temper(LOGITS, bad)
