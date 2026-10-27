"""
Byte-level byte-pair encoding.

This module is the tokenizer the Lecture 1 notes describe. This repository
ships it complete, with the training loop that was Task 1 of Homework 1.

Read the notes before starting. They describe the algorithm in full and trace
three merges on their own sentence; the tests in tests/test_tokenizer.py check
a different hand-traced corpus (hug, pug, pun, bun, hugs).
"""

from collections import Counter


class BPETokenizer:
    """Byte-level byte-pair encoding.

    Learns merge rules from a corpus, then encodes and decodes losslessly.

    Attributes:
        merges: maps a pair of symbol ids (id_a, id_b) to the new id created by
            merging them. Insertion order is the order the rules were learned,
            and that order matters when encoding.
        vocab: maps a symbol id to the byte string it stands for. Ids 0 to 255
            are the raw byte values, so any input at all can be represented
            before a single merge has been learned.
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Pre-tokenization
    # ------------------------------------------------------------------
    @staticmethod
    def _pretokenize(text: str) -> list[list[int]]:
        """Split text into chunks, and return each chunk as a list of byte values.

        Supplied complete. A chunk is one word together with the space before
        it, so "the cat sat" becomes "the", " cat" and " sat". Merges are
        learned only inside a chunk, never across the boundary between two
        words. Because the space stays with the word after it, " the" and
        "the" are different byte sequences and can receive different ids.
        Any other whitespace character becomes a chunk of its own.
        """
        chunks: list[list[int]] = []
        current = ""
        for ch in text:
            if ch.isspace():
                if current:
                    chunks.append(list(current.encode("utf-8")))
                current = ch if ch == " " else ""
                if ch != " ":
                    chunks.append(list(ch.encode("utf-8")))
                    current = ""
            else:
                current += ch
        if current:
            chunks.append(list(current.encode("utf-8")))
        return chunks

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, text: str, num_merges: int) -> None:
        """Learn `num_merges` merge rules from `text`.

        Supplied complete. This was Task 1 of Homework 1, and this repository
        ships the course's implementation of it. On each step the most
        frequent adjacent pair of symbols, counted inside the chunks and never
        across them, becomes a new symbol with the next free id.
        """
        chunks = self._pretokenize(text)

        for step in range(num_merges):
            counts = Counter()
            for chunk in chunks:
                counts.update(zip(chunk, chunk[1:]))

            if not counts:
                break

            best = max(counts, key=counts.get)
            new_id = 256 + step

            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            chunks = [self._apply_merge(chunk, best, new_id) for chunk in chunks]

    @staticmethod
    def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Return a copy of `ids` with every occurrence of `pair` replaced by `new_id`.

        Supplied complete. The same scanning pattern appears in encode().
        """
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids.

        Supplied complete. Merges are applied in the order they were learned,
        which is the order of increasing new id. Recounting pairs in the new
        input and applying whichever pair is currently most frequent is a
        common bug: decoding may still recover the text, but the resulting
        token sequence can differ from the one implied by the trained tokenizer.
        """
        if self.special_tokens:
            import re
            pattern = "(" + "|".join(re.escape(s) for s in
                                     sorted(self.special_tokens, key=len, reverse=True)) + ")"
            out: list[int] = []
            for part in re.split(pattern, text):
                if part in self.special_tokens:
                    out.append(self.special_tokens[part])
                elif part:
                    out.extend(i for chunk in self._pretokenize(part)
                               for i in self._encode_chunk(chunk))
            return out
        ids = [i for chunk in self._pretokenize(text) for i in self._encode_chunk(chunk)]
        return ids

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        """Apply the learned merges, in learned order, to a single chunk.

        Supplied complete.
        """
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            ids = self._apply_merge(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string.

        Supplied complete. Decoding concatenates the byte strings that the ids
        stand for, so it recovers the original text exactly. Special tokens
        decode to their literal strings, because their byte strings are stored
        in the vocabulary like every other id's.
        """
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Special tokens. Complete; you do not write this,
    # but the lecture on instruction tuning is built on it, so read it.
    # ------------------------------------------------------------------
    def add_special_tokens(self, tokens: tuple[str, ...]) -> dict[str, int]:
        """Reserve ids for strings that mark structure rather than text.

        Supplied complete. A special token marks structure the model must be
        able to trust: the boundary between two packed documents, the end of
        a generation, and later the turns of a conversation. Three properties
        follow. The id line below gives the first, and the split at the top
        of encode() gives the other two.

        1. Registered after training, with ids above every merge, so that
           registering a special token never disturbs the learned vocabulary.
        2. Never built by merging. encode() splits the text on the special
           strings before pre-tokenization, so a special token is always
           exactly one token. If ordinary text could merge its way into the
           id for <|endoftext|>, any document could forge a boundary.
        3. The literal string in ordinary text becomes the special id. This
           weakness is left in place on purpose. Production tokenizers refuse
           the string unless the caller allows it, and the lecture on
           instruction tuning returns to why.
        """
        for tok in tokens:
            if tok in self.special_tokens:
                continue
            new_id = 256 + len(self.merges) + len(self.special_tokens)
            self.special_tokens[tok] = new_id
            self.vocab[new_id] = tok.encode("utf-8")
        return dict(self.special_tokens)

    def is_special(self, token_id: int) -> bool:
        return token_id in set(self.special_tokens.values())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """The number of distinct symbols this tokenizer knows about."""
        return len(self.vocab)

    def fertility(self, text: str) -> float:
        """Average number of tokens produced per whitespace-separated word.

        Supplied complete. This is the quantity you measure in the fertility
        experiment. A higher value means the tokenizer spends more tokens on
        the same content.
        """
        words = text.split()
        if not words:
            return 0.0
        return len(self.encode(text)) / len(words)

    def tokens_per_byte(self, text: str) -> float:
        """Tokens produced per byte of UTF-8 encoded input.

        Supplied complete. This is the factor that converts bits per token
        into bits per byte.
        """
        n_bytes = len(text.encode("utf-8"))
        if n_bytes == 0:
            return 0.0
        return len(self.encode(text)) / n_bytes
