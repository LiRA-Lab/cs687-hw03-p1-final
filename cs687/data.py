"""
Turning a stream of token ids into training pairs.

Autoregressive training needs pairs of the form (input, target) where the
target is the input shifted forward by one position. One forward pass through
the model then produces a classification loss at every position at once, which
is why the chain-rule factorization the notes derive costs nothing extra at
training time.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class NextTokenDataset(Dataset):
    """Cuts a long stream of token ids into overlapping (input, target) windows.

    For a window starting at position s with context length T:

        input  = ids[s     : s + T]        that is, (y_s,   ..., y_{s+T-1})
        target = ids[s + 1 : s + T + 1]    that is, (y_{s+1}, ..., y_{s+T})

    Args:
        ids: the full stream of token ids, as a plain Python list.
        context_len: how many tokens the model sees at once.
        stride: how far to move the window between consecutive samples.
            A stride smaller than the context length produces overlapping
            windows: more training pairs from the same text, at the cost of
            correlation between them. A stride equal to the context length
            covers the text exactly once with no overlap.
    """

    def __init__(self, ids: list[int], context_len: int, stride: int):
        self.inputs: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        # Supplied complete. This loop was Task 2 of Homework 1. The last
        # window needs one extra token for its target, so the walk stops at
        # len(ids) - context_len.
        for start in range(0, len(ids) - context_len, stride):
            input_ids = ids[start : start + context_len]
            target_ids = ids[start + 1 : start + context_len + 1]
            self.inputs.append(torch.tensor(input_ids, dtype=torch.long))
            self.targets.append(torch.tensor(target_ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[i], self.targets[i]


def make_loader(
    ids: list[int],
    context_len: int = 256,
    stride: int = 128,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap NextTokenDataset in a DataLoader. Supplied complete."""
    dataset = NextTokenDataset(ids, context_len, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
