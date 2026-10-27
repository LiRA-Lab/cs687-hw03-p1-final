# CS 687 · Homework 3 · Project 1, final checkpoint

This is a standalone student repository for Homework 3. It trains the
decoder-only model of Homework 2 on a text of Shakespeare's plays, reads the
run from its two losses, resumes it from a checkpoint, and decodes text from
the trained model under temperature, top-p and top-k. Project 1 ends here.

The repository ships the complete Homework 1 and Homework 2 code, the tokenizer,
the dataset, the input embedding, attention, the block and the model, with
the course's answers inserted. A graphics card is not required, but the
training run takes a few minutes on any Colab GPU and much longer on the CPU
runtime, so a GPU runtime is recommended.

## What you must complete

The Colab notebook contains twelve assessed answer cells:

- four short guided-calculation cells;
- four programming-task cells; and
- four Markdown report-response cells.

The programming tasks are:

| task | tagged notebook cell | function |
|---|---|---|
| Task 1 | `answer-task-1` | `train_step` |
| Task 2 | `answer-task-2` | `apply_temperature` |
| Task 3 | `answer-task-3` | `top_p_filter` |
| Task 4 | `answer-task-4` | `top_k_filter` |

All eight code-answer cells contain `NotImplementedError` placeholders. The
surrounding code is supplied complete: the schedule, the checkpoint's saving
and loading, the training loop, the held-out loss, the sampler and the
generation loop. The four report cells contain written response
placeholders. Your completed notebook is the authoritative submission; you do
not need to copy its implementations into `.py` files.

## Grading

Homework 3 is graded out of 100 points:

| assessed work | points |
|---|---:|
| Four guided calculations | 16 |
| Task 1: one step of training | 10 |
| Task 2: temperature | 3 |
| Task 3: top-p | 7 |
| Task 4: top-k | 4 |
| Four written responses | 60 |
| **Total** | **100** |

All eight code exercises are graded all-or-nothing. Each guided calculation
receives either 4 points or 0 points. Each task receives its points only if it
passes every staff-controlled test of that task. Partially correct code does
not receive partial credit. The written responses are graded separately.

The training run itself is not graded. Its outputs, the printed losses, the
plot of the two curves and the samples, are the evidence your written
responses are read against, so save the notebook with its outputs. Grading
runs your notebook again from a clean environment to check the calculations
and the four tasks, with a short training run in place of yours.

## Google Colab

The notebook, `homework03_colab.ipynb`, is attached to the Moodle assignment
for Homework 3. Download it from there and upload it to Google Colab. Work in
that uploaded notebook and save it there; do not create a new notebook and
paste the cells into it, because a new notebook loses the cell identifiers and
metadata that grading depends on. Choose a GPU runtime under Runtime > Change
runtime type. Run the notebook from top to bottom. Its setup cell clones this
repository and installs the required Python packages. After setup the notebook
needs no network access.

The notebook lets you develop all four implementations interactively. A
focused check follows each code-answer cell so that you receive feedback
before continuing. Near the end, a final cell runs all 30 public tests
together. Run these cells inside the notebook: a separate `pytest` process
cannot see definitions that exist only in the notebook kernel.

The training run of section 4 writes a checkpoint, and the supplied loop
resumes from it when the notebook is run again, so running the notebook from
the top a second time restores the finished run in seconds as long as the
session's disk is still there. Delete `runs/checkpoint.pt` to train again
from scratch.

## Homework workflow

1. Read the second part of the Lecture 2 notes.
2. Work through the notebook in order. Complete all four guided-calculation
   cells and the four programming-task cells, and run each immediate check.
3. Restart the runtime and run the complete notebook from top to bottom.
4. Confirm that the final check runs all four calculation checks and reports
   that all 30 public tests pass.
5. Complete the four questions in the notebook's report section, using
   evidence from your own run and samples.

Do not modify the tests to make an implementation pass. The tests describe the
required behavior and grading uses a staff-controlled copy. Write the four
functions yourself: the point of the homework is to see what is inside the
loop and the sampler.

## Protect the notebook structure

Automatic grading depends on the released notebook structure. Follow these
rules while working:

- In a code-answer cell, edit only between `YOUR CODE STARTS HERE` and
  `YOUR CODE ENDS HERE`, and keep both marker lines.
- In a written-answer cell, replace only the placeholder below `**Response:**`.
  Do not remove the response marker.
- Do not delete, duplicate, or reorder supplied cells. You may add separate
  scratch cells for your own experiments. Grading removes every cell that is
  not part of the released notebook before it runs your notebook, so a scratch
  cell never counts: each answer must work on its own inside its answer
  region, with every helper and import it needs written there.
- Do not convert a supplied code cell to Markdown or a supplied Markdown cell
  to code.
- Do not change notebook metadata, cell tags, stable cell IDs, the assignment
  identifier, or the assignment-version number.
- Do not edit supplied instructions, demonstrations, setup code, checks, or
  code outside marked answer regions.
- Keep the outputs of the training run and the samples in the saved notebook,
  and leave Colab's setting "Omit code cell output when saving this notebook"
  off.

Execution counts do not matter, because grading runs your notebook again from
a fresh kernel.

If you accidentally change the structure, obtain a fresh copy of the released
notebook and copy only your answers into its designated answer regions. A
submission whose structure has changed is set aside for manual inspection and
loses 20 points, because staff have to repair the file by hand before it can
be graded. An answer that cannot be identified reliably receives zero for that
component.

## Deliverables

- the completed Homework 3 notebook, with its outputs

The completed notebook is the single Homework 3 submission. Save the notebook
in Colab, rename it to `homework03_colab_STUDENTNUMBER.ipynb`, replacing
`STUDENTNUMBER` with your student number, for example
`homework03_colab_21802962.ipynb`, and download it with File > Download >
Download .ipynb. Upload that exact file, and only that file, to the Moodle
assignment for Homework 3. Moodle's account record, rather than the filename
alone, remains the authoritative student identity. The required filename
format is nevertheless mandatory, and a submission whose file name does not
follow it loses 10 points.

A late submission loses 10 points for every hour past the submission time,
and a started hour counts as a whole hour, so a submission ten hours late
receives zero. Moodle's record of the upload time is what counts.

You may submit unanswered work; an unanswered cell receives zero for that
component. An unreadable file or a file that is not an `.ipynb` notebook
cannot be graded and receives a grade of zero.

## Repository layout

```text
README.md            this file
requirements.txt     the Python packages the notebook installs
pytest.ini           the test configuration the notebook's checks use
notebook_support.py  supplied setup, calculation checks, plotting and public-test helpers
cs687/               the course package: Homeworks 1 and 2 complete, the four Homework 3 tasks left for you
tests/               inspectable public tests run by the notebook, one file per task
data/                the training text, a collection of Shakespeare's plays, public domain
notebooks/           the Colab entry point and authoritative code submission
```
