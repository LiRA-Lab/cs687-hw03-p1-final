"""Tests for Task 1: train_step.

The notebook runs these for you. Each test reaches the function through the
module, so that the version the notebook attached is the one tested. They use
a tiny stand-in model so that they run in a moment; the loop in the notebook
runs the same function on the real one.
"""

import copy

import torch
import torch.nn as nn

from cs687 import training


def step(*args, **kwargs):
    return training.train_step(*args, **kwargs)


class Tiny(nn.Module):
    """A two-layer stand-in for the model: ids in, one score per id out."""

    def __init__(self, vocab=10, width=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.out = nn.Linear(width, vocab)

    def forward(self, x):
        return self.out(self.emb(x))


def batch(seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, 10, (4, 6)), torch.randint(0, 10, (4, 6))


def test_returns_the_loss_as_a_plain_number():
    """The step returns a float, not a tensor, for the loop's log."""
    torch.manual_seed(0)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = step(model, optimizer, *batch())
    assert isinstance(loss, float), (
        f"The step should return a plain Python float, but returned {type(loss).__name__}."
    )
    assert 0.0 < loss < 10.0, f"The loss {loss} is not a plausible cross-entropy for ten ids."


def test_the_returned_loss_is_the_cross_entropy_on_the_batch():
    """The number returned is the mean cross-entropy over all B*T slots, before the update."""
    torch.manual_seed(0)
    model = Tiny()
    frozen = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs, targets = batch()
    with torch.no_grad():
        logits = frozen(inputs)
        expected = nn.functional.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1)).item()
    loss = step(model, optimizer, inputs, targets)
    assert abs(loss - expected) < 1e-6, (
        f"The step returned {loss:.6f}, but the mean cross-entropy over the 24 slots of the "
        f"batch is {expected:.6f}. The loss is averaged over every slot of every window."
    )


def test_the_loss_falls_when_one_batch_is_repeated():
    """Training on the same batch again and again must drive its loss down."""
    torch.manual_seed(0)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    inputs, targets = batch()
    first = step(model, optimizer, inputs, targets)
    for _ in range(60):
        last = step(model, optimizer, inputs, targets)
    assert last < first * 0.5, (
        f"The loss went from {first:.3f} to {last:.3f} over sixty steps on one batch. It should "
        "fall sharply, so the update is probably not moving the parameters against the gradient."
    )


def test_no_gradient_is_left_behind_after_the_step():
    """The step ends with the stored gradients zero, or cleared, for the next step."""
    torch.manual_seed(0)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step(model, optimizer, *batch())
    for parameter in model.parameters():
        assert parameter.grad is None or torch.all(parameter.grad == 0), (
            "A gradient survived the step. The framework adds each new gradient to the stored "
            "one, so a step that does not set the gradients to zero uses the sum of every "
            "gradient so far."
        )


def test_the_gradient_is_clipped_before_the_update():
    """With a tiny clipping bound the parameters barely move, whatever the learning rate."""
    torch.manual_seed(0)
    model = Tiny()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    step(model, optimizer, *batch(), clip_norm=1e-6)
    moved = max(
        (after - earlier).abs().max().item()
        for after, earlier in zip(model.parameters(), before)
    )
    assert moved < 1e-4, (
        f"The parameters moved by {moved:.2e} although the gradient was clipped to a norm of "
        "1e-6. The clipping must act on the gradient the update uses, so it comes after the "
        "backward pass and before the optimizer's step."
    )


def test_the_clipping_scales_the_whole_gradient_by_one_factor():
    """Clipping by norm keeps the gradient's direction: with a bound of a tenth of its norm, every entry shrinks tenfold."""
    torch.manual_seed(0)
    model = Tiny()
    frozen = copy.deepcopy(model)
    inputs, targets = batch()
    nn.functional.cross_entropy(frozen(inputs).reshape(-1, 10), targets.reshape(-1)).backward()
    gradient = torch.cat([parameter.grad.flatten() for parameter in frozen.parameters()])
    bound = 0.1 * gradient.norm().item()
    before = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    step(model, optimizer, inputs, targets, clip_norm=bound)
    change = before - torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    assert torch.allclose(change, 0.1 * gradient, rtol=1e-3, atol=1e-7), (
        "With plain gradient descent at learning rate one and a bound of a tenth of the "
        "gradient's norm, every parameter should move by a tenth of its gradient entry. The "
        "clipping scales all the gradients together by one factor, so that their joint norm "
        "equals the bound; clipping each entry on its own changes the gradient's direction."
    )
