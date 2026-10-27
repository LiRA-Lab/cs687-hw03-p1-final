"""Operational helpers used by the Homework 3 student notebook.

The notebook keeps the assessed code and the demonstrations visible. This
module holds only the setup, the checks of the guided calculations, the
plotting of the loss curves, and the running of the public tests, none of
which students are asked to write.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


def prepare_notebook(repo_root: Path) -> Path:
    """Enter the repository, install its requirements, and make it importable."""
    repo_root = repo_root.resolve()
    required = (repo_root / "cs687", repo_root / "tests", repo_root / "requirements.txt")
    if not required[0].is_dir() or not required[1].is_dir() or not required[2].is_file():
        raise RuntimeError(
            "The notebook setup did not locate a Homework 3 repository containing "
            "cs687/, tests/, and requirements.txt."
        )

    os.chdir(repo_root)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )
    root_text = str(repo_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    # The three-step standard of the checkpoint asks for the same losses to
    # the last digit, and on a graphics card that needs the deterministic
    # kernels, which this setting selects before the first CUDA call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print("ready")
    return repo_root


def pick_device() -> "torch.device":
    """A CUDA device when the runtime has one, otherwise the processor."""
    import torch

    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)
        return torch.device("cuda")
    return torch.device("cpu")


def _is_incomplete(function: Callable[..., object]) -> bool:
    """Return whether an answer still contains its supplied placeholder."""
    return "NotImplementedError" in function.__code__.co_names


def _close(value: object, expected: float, tolerance: float) -> bool:
    try:
        return abs(float(value) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# Checks of the four guided calculations
# ----------------------------------------------------------------------

def check_starting_loss(
    vocab_size: int, tokens_per_byte: float, untrained_loss: float,
    start_loss_nats: object, start_perplexity: object,
    start_bits_per_token: object, start_bits_per_byte: object,
) -> None:
    """Check the loss of a model that predicts every id with the same probability, in four units, and the measured loss against it."""
    nats = math.log(vocab_size)
    if not _close(start_loss_nats, nats, 1e-3):
        raise AssertionError(
            "The loss of a model that predicts every id with the same probability is the logarithm "
            "of the vocabulary size, "
            "in nats, since every id then has probability one over V."
        )
    if not _close(start_perplexity, vocab_size, 1e-2):
        raise AssertionError(
            "The perplexity is the exponential of the loss in nats, which for that model is "
            "the vocabulary size itself."
        )
    if not _close(start_bits_per_token, nats / math.log(2), 1e-3):
        raise AssertionError(
            "Bits per token is the loss in nats divided by the logarithm of two."
        )
    if not _close(start_bits_per_byte, nats / math.log(2) * tokens_per_byte, 1e-3):
        raise AssertionError(
            "Bits per byte is bits per token times the tokens per byte of the tokenizer on the text."
        )
    print("The starting loss and its three other readings match.")
    gap = float(untrained_loss) - nats
    if abs(gap) > 1.0:
        raise AssertionError(
            f"The untrained model's held-out loss is {float(untrained_loss):.3f} nats per token, "
            f"{gap:+.3f} from log V = {nats:.3f}. A loss far above log V before training points to "
            "a fault in the model, its initialization or the loss calculation, and one far "
            "below it to a fault in the loss calculation or to a model that did not start "
            "from scratch."
        )
    print(
        f"The untrained model's held-out loss, {float(untrained_loss):.3f}, is {gap:+.3f} from "
        f"log V = {nats:.3f}, as nearly equal initial scores should give."
    )


def check_run_budget(
    steps: int, batch_size: int, context_len: int, parameter_count: int, text_tokens: int,
    tokens_seen: object, passes_over_text: object, reference_tokens: object,
    factor_below_reference: object,
) -> None:
    """Check the tokens the run sees against the reference of twenty per parameter."""
    seen = steps * batch_size * context_len
    reference = 20 * parameter_count
    if not _close(tokens_seen, seen, 0.5):
        raise AssertionError(
            "The tokens a run sees are the steps times the windows per batch times the slots per window."
        )
    if not _close(passes_over_text, seen / text_tokens, 1e-2):
        raise AssertionError(
            "The passes over the training text are the tokens seen divided by the tokens of the text."
        )
    if not _close(reference_tokens, reference, 0.5):
        raise AssertionError(
            "The reference of the notes is twenty tokens for every parameter, so its tokens are "
            "twenty times the parameter count."
        )
    if not _close(factor_below_reference, reference / seen, 1e-2):
        raise AssertionError(
            "The factor by which the run falls below the reference is the reference's tokens "
            "divided by the tokens seen."
        )
    print("The budget of the run matches.")


def check_temperature_table(
    logits: "torch.Tensor",
    favourite_at_half: object, favourite_at_three_halves: object,
    tail_at_half: object, tail_at_three_halves: object,
) -> None:
    """Check the favourite and the tail of the notes' table at two temperatures."""
    import torch

    def favourite(tau: float) -> float:
        return torch.softmax(logits / tau, dim=-1)[0].item()

    def tail(tau: float) -> float:
        return torch.softmax(logits / tau, dim=-1)[-4:].sum().item()

    if not _close(favourite_at_half, favourite(0.5), 2e-3):
        raise AssertionError(
            "At temperature 0.5 every logit is divided by 0.5 before the softmax; the favourite "
            "should take about 0.84 of the mass."
        )
    if not _close(favourite_at_three_halves, favourite(1.5), 2e-3):
        raise AssertionError(
            "At temperature 1.5 every logit is divided by 1.5 before the softmax; the favourite "
            "should fall to about 0.40."
        )
    if not _close(tail_at_half, tail(0.5), 2e-3):
        raise AssertionError(
            "The tail is the mass of the last four ids of the table after the softmax at that temperature."
        )
    if not _close(tail_at_three_halves, tail(1.5), 2e-3):
        raise AssertionError(
            "The tail is the mass of the last four ids of the table after the softmax at that temperature."
        )
    print("The favourite and the tail at both temperatures match.")


def show_temperatures(logits: "torch.Tensor", tokens: list[str]) -> None:
    """Print the notes' table at three temperatures, with the mass of the two alternatives and of the last four ids."""
    import torch

    temperatures = (0.5, 1.0, 1.5)
    columns = [torch.softmax(logits / tau, dim=-1) for tau in temperatures]
    print()
    print(f"{'':>12}" + "".join(f"{'tau = ' + format(tau, 'g'):>11}" for tau in temperatures))
    for index, token in enumerate(tokens):
        print(f"{token:>12}" + "".join(f"{column[index].item():>11.3f}" for column in columns))
    print(f"{tokens[1] + ' and ' + tokens[2]:>12}" + "".join(f"{column[1:3].sum().item():>11.3f}" for column in columns))
    print(f"{'last four':>12}" + "".join(f"{column[-4:].sum().item():>11.3f}" for column in columns))


def check_two_cuts(
    logits: "torch.Tensor",
    cumulative_mass: object, kept_at_p_09: object,
    favourite_under_top_p: object, favourite_under_top_k: object,
) -> None:
    """Check the running sums of the table and the two renormalized favourites."""
    import torch

    probabilities = torch.softmax(logits, dim=-1)
    running = probabilities.cumsum(dim=-1)
    try:
        given = torch.as_tensor(cumulative_mass, dtype=torch.float).flatten()
    except (TypeError, ValueError):
        given = torch.empty(0)
    if given.numel() != running.numel() or not torch.allclose(given, running, atol=2e-3):
        raise AssertionError(
            "The cumulative mass is the running sum of the probabilities of the table, from the "
            "most probable id down, one entry per id."
        )
    count = int((running - probabilities < 0.9).sum().item())
    if not _close(kept_at_p_09, count, 0.5):
        raise AssertionError(
            "Top-p at 0.9 keeps the ids from the most probable down until their mass reaches 0.9, "
            "including the id that carries the sum past it."
        )
    if not _close(favourite_under_top_p, (probabilities[0] / running[count - 1]).item(), 2e-3):
        raise AssertionError(
            "Under top-p the kept probabilities are scaled up to sum to one, so the favourite's "
            "probability is divided by the mass of the kept set."
        )
    if not _close(favourite_under_top_k, (probabilities[0] / running[2]).item(), 2e-3):
        raise AssertionError(
            "Under top-k at 3 the three most probable ids are kept and scaled up to sum to one."
        )
    print("The cumulative mass, the kept count and the two renormalized favourites match.")


# ----------------------------------------------------------------------
# The plot of the two losses
# ----------------------------------------------------------------------

def plot_loss_curves(curve: list[dict], vocab_size: int) -> None:
    """Draw the training and the held-out loss of a run against the step."""
    import matplotlib.pyplot as plt

    steps = [point["step"] for point in curve]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, [point["train"] for point in curve], label="training loss", color="#1f3b73")
    axis.plot(steps, [point["held_out"] for point in curve], label="held-out loss", color="#c0392b")
    axis.axhline(math.log(vocab_size), color="grey", linestyle="--", linewidth=1, label="log V")
    axis.set_xlabel("step")
    axis.set_ylabel("loss, nats per token")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Attaching the answers and running the public tests
# ----------------------------------------------------------------------

def _run_pytest(path: str, failure_message: str) -> None:
    """Run one public-test target in the current notebook kernel."""
    import pytest

    # A test module imported by an earlier run would keep the earlier answer,
    # so the test modules are reloaded on every run.
    for name in list(sys.modules):
        if name.startswith("test_"):
            del sys.modules[name]

    result = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", path])
    if result != pytest.ExitCode.OK:
        raise AssertionError(failure_message)


def attach_train_step(train_step: Callable[..., object]) -> None:
    """Make the notebook's training step the one the package and the loop use."""
    import cs687
    import cs687.training as training_module

    training_module.train_step = train_step
    cs687.train_step = train_step


def attach_temperature(apply_temperature: Callable[..., object]) -> None:
    import cs687
    import cs687.sampling as sampling_module

    sampling_module.apply_temperature = apply_temperature
    cs687.apply_temperature = apply_temperature


def attach_top_p(top_p_filter: Callable[..., object]) -> None:
    import cs687
    import cs687.sampling as sampling_module

    sampling_module.top_p_filter = top_p_filter
    cs687.top_p_filter = top_p_filter


def attach_top_k(top_k_filter: Callable[..., object]) -> None:
    import cs687
    import cs687.sampling as sampling_module

    sampling_module.top_k_filter = top_k_filter
    cs687.top_k_filter = top_k_filter


def _require_complete(pairs: list[tuple[str, Callable[..., object]]]) -> None:
    incomplete = [name for name, function in pairs if _is_incomplete(function)]
    if incomplete:
        raise AssertionError(
            "Complete and rerun these answer cells first: " + ", ".join(incomplete)
        )


def run_task1_tests(train_step: Callable[..., object]) -> None:
    """Attach the Task 1 answer and run its public tests."""
    if _is_incomplete(train_step):
        raise AssertionError(
            "Task 1 is still incomplete. Replace the NotImplementedError in the "
            "answer-task-1 cell, run that cell again, and rerun this check."
        )
    attach_train_step(train_step)
    _run_pytest(
        "tests/test_train_step.py",
        "Task 1 has not passed all public tests. Review the failures above.",
    )
    print("Task 1 passed all public tests.")


def run_task2_tests(apply_temperature: Callable[..., object]) -> None:
    """Attach the Task 2 answer and run its public tests."""
    _require_complete([("Task 2", apply_temperature)])
    attach_temperature(apply_temperature)
    _run_pytest(
        "tests/test_temperature.py",
        "Task 2 has not passed all public tests. Review the failures above.",
    )
    print("Task 2 passed all public tests.")


def run_task3_tests(top_p_filter: Callable[..., object]) -> None:
    """Attach the Task 3 answer and run its public tests. Needs Task 2."""
    _require_complete([("Task 3", top_p_filter)])
    attach_top_p(top_p_filter)
    _run_pytest(
        "tests/test_top_p.py",
        "Task 3 has not passed all public tests. Review the failures above.",
    )
    print("Task 3 passed all public tests.")


def run_task4_tests(top_k_filter: Callable[..., object]) -> None:
    """Attach the Task 4 answer and run its public tests. Needs Task 2."""
    _require_complete([("Task 4", top_k_filter)])
    attach_top_k(top_k_filter)
    _run_pytest(
        "tests/test_top_k.py",
        "Task 4 has not passed all public tests. Review the failures above.",
    )
    print("Task 4 passed all public tests.")


def run_all_public_tests(
    train_step: Callable[..., object],
    apply_temperature: Callable[..., object],
    top_p_filter: Callable[..., object],
    top_k_filter: Callable[..., object],
) -> None:
    """Attach all four answers and run the complete public test suite."""
    _require_complete([
        ("Task 1", train_step),
        ("Task 2", apply_temperature),
        ("Task 3", top_p_filter),
        ("Task 4", top_k_filter),
    ])
    attach_train_step(train_step)
    attach_temperature(apply_temperature)
    attach_top_p(top_p_filter)
    attach_top_k(top_k_filter)
    _run_pytest(
        "tests",
        "One or more public tests failed. Review the failures above.",
    )
    print("All 30 public tests passed.")
