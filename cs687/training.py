"""
Training: one step, the loop that repeats it, and the checkpoint.

This module is the training of the second part of the Lecture 2 notes. One
step of training is five lines, the loop repeats the step over batches and
reads the run from its two losses, and a checkpoint lets an interrupted run
continue as the same run. The step is left for you to write, and the
notebook attaches your version to this module in place of the stub.
Everything else is supplied complete, and the loop calls your step.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from cs687.data import NextTokenDataset


# ----------------------------------------------------------------------
# Task 1: one step of training
# ----------------------------------------------------------------------
def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    clip_norm: float = 1.0,
) -> float:
    """Run one step of training on one batch and return its loss as a number.

    The notes' algorithm "one training step", in order: the forward pass
    gives the logits of shape (B, T, V); the loss is the cross-entropy between
    the logits and the targets with the B*T slots flattened into one list; the
    backward pass fills the gradient of every parameter; the gradient is
    clipped so that its norm is at most clip_norm; the optimizer moves every
    parameter; and the stored gradients are set to zero.

    Args:
        model: the model, already in training mode.
        optimizer: the optimizer whose step moves the parameters.
        inputs: token ids, shape (B, T).
        targets: the ids that follow, shape (B, T).
        clip_norm: the bound on the norm of the gradient.

    Returns:
        The loss on the batch as a plain Python float.

    Library pieces, and what each takes and does: calling the model on the
    inputs runs the forward pass and returns the logits of shape (B, T, V);
    a tensor's reshape method lays its entries out in a new shape, with -1
    for an axis it should work out, and the loss function wants the logits
    as a matrix with one row per scored slot and V columns and the targets
    as a vector with one id per row; torch.nn.functional.cross_entropy takes
    that matrix and that vector and returns the mean over the rows of minus
    the log-probability of each row's target, as a one-element tensor;
    loss.backward() runs the backward pass and adds each parameter's gradient
    into its .grad field; torch.nn.utils.clip_grad_norm_(parameters, bound)
    scales all the stored gradients down in place when their joint norm
    exceeds the bound; optimizer.step() moves the parameters using the
    stored gradients and optimizer.zero_grad() clears them; and a
    one-element tensor's item() returns its value as a Python number.
    """
    raise NotImplementedError(
        "Task 1 is written in the homework notebook, which attaches it to this module."
    )


# ----------------------------------------------------------------------
# Supplied: the learning-rate schedule
# ----------------------------------------------------------------------
def learning_rate_at(step: int, peak: float, warmup_steps: int, total_steps: int,
                     floor_fraction: float = 0.1) -> float:
    """The learning rate of one step: linear warmup, then cosine decay to a floor.

    Supplied complete. The notes name this schedule as one line of the loop.
    The rate rises from near zero to the peak over the first warmup_steps,
    then falls along a cosine to floor_fraction of the peak at the last step.
    """
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    floor = peak * floor_fraction
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------
# Supplied: the random-number generator's state, on every device in use
# ----------------------------------------------------------------------
def get_generator_state() -> dict:
    """The state of the random-number generators, as one object a checkpoint can hold.

    Supplied complete. The batches are drawn on the processor, and dropout
    draws on the device the model runs on, so both generators are part of
    the state of a run.
    """
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_generator_state(state: dict) -> None:
    """Restore what get_generator_state returned. Supplied complete."""
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ----------------------------------------------------------------------
# Supplied: the checkpoint, its saving and its loading
# ----------------------------------------------------------------------
def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, extra: dict | None = None) -> None:
    """Write everything a resumed run needs to continue as the same run.

    Supplied complete. The file holds the four things the notes list: the
    parameters, the optimizer's state, the step counter, and the state of
    the random-number generator. `extra` carries whatever else the loop wants
    back, such as the loss curve so far. The file is written under a
    temporary name and then renamed into place, so that a job killed in the
    middle of a save leaves the old file or the new one and never a
    half-written one.

    The keys of the saved dictionary are "model", "optimizer", "step", "rng"
    and "extra"; load_checkpoint reads them back.
    """
    path = Path(path)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "rng": get_generator_state(),
        "extra": extra or {},
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer) -> tuple[int, dict]:
    """Restore a checkpoint written by save_checkpoint, and return the step
    the run continues from and the saved extra dictionary.

    Supplied complete. It is the notes' algorithm "resume a run from a
    checkpoint": read the file, load the saved parameters into the model and
    the saved state into the optimizer, restore the state of the
    random-number generator, and return the saved step counter plus one,
    since the saved step is the one that had just finished, together with
    the extra dictionary.

    The standard this meets is the notes' three-step standard: on one
    machine, a run that trains three steps, saves, and trains three more
    gives the same three losses, to the last digit, as a fresh model and
    optimizer that load the file and train three steps. The notebook shows
    it on your own run, and shows what happens without the optimizer's state.

    Args:
        path: the checkpoint file.
        model: a freshly built model of the same configuration.
        optimizer: a freshly built optimizer over that model's parameters.

    Returns:
        (next_step, extra).
    """
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    set_generator_state(saved["rng"])
    return saved["step"] + 1, saved.get("extra", {})


# ----------------------------------------------------------------------
# Supplied: the batches, the held-out loss, and the loop
# ----------------------------------------------------------------------
def sample_batch(dataset: NextTokenDataset, batch_size: int,
                 device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Draw batch_size windows of the dataset at random. Supplied complete.

    The draw uses the processor's generator whatever device the model runs
    on, so the generator state a checkpoint restores is the state that
    decides the batches.
    """
    index = torch.randint(len(dataset), (batch_size,))
    inputs = torch.stack([dataset.inputs[i] for i in index.tolist()])
    targets = torch.stack([dataset.targets[i] for i in index.tolist()])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def held_out_loss(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """The mean loss on a fixed list of held-out batches, in evaluation mode. Supplied complete."""
    was_training = model.training
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        logits = model(inputs)
        total += nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        ).item()
    if was_training:
        model.train()
    return total / len(batches)


def train_run(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: NextTokenDataset,
    held_out_batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    batch_size: int,
    peak_learning_rate: float,
    warmup_steps: int,
    checkpoint_path: str | Path,
    evaluate_every: int = 100,
    checkpoint_every: int = 500,
    clip_norm: float = 1.0,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """The training loop of the notes. Supplied complete; it calls your train_step.

    Every pass of the loop sets the learning rate for the step, draws a batch,
    runs train_step and records the loss. Every evaluate_every steps it
    measures the held-out loss and prints both, and every checkpoint_every
    steps it writes the checkpoint with the curve so far.

    Before the first step it prints log V, the loss of a model that predicts
    every id with the same probability, which the first printed loss should
    be near. A first loss far above it points to a fault in the model, its
    initialization or the loss calculation, and one far below it to a fault
    in the loss calculation or to a run that did not start from scratch. A
    model that can see its targets still starts near log V, because its
    random parameters cannot copy them yet, and the leak shows once training
    begins, as a loss that falls almost to zero.

    If the checkpoint file already exists, the loop resumes from it through
    load_checkpoint, so that a run interrupted by the end of a session goes
    on as the same run, and a run that already finished is restored in
    seconds instead of trained again. Delete the file to train from scratch.

    The environment variable CS687_TRAINING_STEPS, when set, replaces steps.
    The grader sets it so that checking a notebook does not repeat the whole
    run; the run you report is the one saved in your notebook.

    Returns the curve: one dictionary per evaluation with the step, the
    training loss of that step, the held-out loss, and the learning rate.
    """
    steps = int(os.environ.get("CS687_TRAINING_STEPS", steps))
    checkpoint_path = Path(checkpoint_path)
    vocab_size = model.config["vocab_size"]
    print(f"log V = log {vocab_size} = {math.log(vocab_size):.3f} nats per token, the loss of a model that predicts every id with the same probability")

    curve: list[dict] = []
    start = 0
    if checkpoint_path.exists():
        start, extra = load_checkpoint(checkpoint_path, model, optimizer)
        curve = list(extra.get("curve", []))
        print(f"resumed from {checkpoint_path} at step {start}: the same run, continued")
        if start >= steps:
            print("the run had already finished; nothing left to train")
            return curve

    print(f"\n{'step':>6} {'train':>8} {'held-out':>9} {'lr':>9}")
    started = time.time()
    model.train()
    for step in range(start, steps):
        rate = learning_rate_at(step, peak_learning_rate, warmup_steps, steps)
        for group in optimizer.param_groups:
            group["lr"] = rate
        inputs, targets = sample_batch(train_dataset, batch_size, device)
        loss = train_step(model, optimizer, inputs, targets, clip_norm)

        last = step == steps - 1
        if step % evaluate_every == 0 or last:
            held_out = held_out_loss(model, held_out_batches)
            curve.append({"step": step, "train": round(loss, 4), "held_out": round(held_out, 4), "lr": rate})
            print(f"{step:>6} {loss:>8.3f} {held_out:>9.3f} {rate:>9.2e}")
        if (checkpoint_every and step > 0 and step % checkpoint_every == 0) or last:
            save_checkpoint(checkpoint_path, model, optimizer, step, extra={"curve": curve})

    print(f"\ntrained {steps - start} steps in {time.time() - started:.0f} s; checkpoint at {checkpoint_path}")
    return curve
