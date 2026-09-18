"""SentencePiece-style unigram tokenizer used by the Gist model."""

from __future__ import annotations

import struct
import unicodedata
from dataclasses import dataclass

_METASPACE = "\u2581"
_CONTROL = (
    set(range(0x01, 0x09))
    | {0x0B}
    | set(range(0x0E, 0x20))
    | {
        0x7F,
        0x8F,
        0x9F,
    }
)
_SPACES = {
    0x09,
    0x0A,
    0x0C,
    0x0D,
    0xA0,
    0x1680,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x2581,
    0x3000,
    0xFEFF,
    0xFFFD,
} | set(range(0x2000, 0x2010))


@dataclass(frozen=True)
class Token:
    """A token ID and the normalized Unicode scalars it covers."""

    id: int
    scalars: tuple[str, ...]


def nmt_normalize(text: str) -> str:
    """Apply the SentencePiece ``nmt_nfkc`` normalization used by Gist."""

    pre_normalized = "".join(
        " " if ord(character) in _SPACES else character
        for character in text
        if ord(character) not in _CONTROL
    )
    if "\uff5e" not in pre_normalized:
        return unicodedata.normalize("NFKC", pre_normalized)
    return "\uff5e".join(
        unicodedata.normalize("NFKC", segment)
        for segment in pre_normalized.split("\uff5e")
    )


class Tokenizer:
    """Parse and run the compact Gist unigram tokenizer."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 21 or data[:4] != b"GSTK":
            raise ValueError("invalid tokenizer header")

        offset = 5
        unk_id, bos_id, eos_id, count = struct.unpack_from("<4i", data, offset)
        offset += 16
        if count <= 0 or count > (len(data) - offset) // 6:
            raise ValueError("invalid tokenizer vocabulary count")

        score_bytes = count * 4
        if offset + score_bytes > len(data):
            raise ValueError("truncated tokenizer scores")
        scores = list(struct.unpack_from(f"<{count}f", data, offset))
        offset += score_bytes

        length_bytes = count * 2
        if offset + length_bytes > len(data):
            raise ValueError("truncated tokenizer piece lengths")
        lengths = struct.unpack_from(f"<{count}H", data, offset)
        offset += length_bytes

        pieces: dict[str, int] = {}
        maximum_length = 1
        for piece_id, length in enumerate(lengths):
            end = offset + length
            if end > len(data):
                raise ValueError("truncated tokenizer piece")
            piece = data[offset:end].decode("utf-8", errors="replace")
            offset = end
            pieces[piece] = piece_id
            maximum_length = max(maximum_length, len(piece))

        if offset != len(data):
            raise ValueError("unexpected trailing tokenizer data")
        if not 0 <= unk_id < count:
            raise ValueError("tokenizer unknown ID is out of range")

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self._scores = scores
        self._pieces = pieces
        self._max_length = min(maximum_length, 32)
        self._unknown_penalty = min(scores) - 10.0

    def encode(self, text: str) -> list[int]:
        """Return content-subword IDs for ``text``."""

        return [token.id for token in self.tokenize(text)]

    def tokenize(self, text: str) -> list[Token]:
        """Return Viterbi-optimal tokens for ``text``."""

        normalized = nmt_normalize(text)
        squeezed: list[str] = []
        last_was_space = True
        for character in normalized:
            if character == " ":
                if last_was_space:
                    continue
                last_was_space = True
            else:
                last_was_space = False
            squeezed.append(character)
        if squeezed and squeezed[-1] == " ":
            squeezed.pop()

        scalar_text = [
            _METASPACE if character == " " else character for character in squeezed
        ]
        scalar_text.insert(0, _METASPACE)
        length = len(scalar_text)
        if length == 0:
            return []

        best = [-1e18] * (length + 1)
        best[0] = 0.0
        back_positions = [-1] * (length + 1)
        back_ids = [-1] * (length + 1)

        for end in range(1, length + 1):
            start_limit = max(0, end - self._max_length)
            for start in range(start_limit, end):
                piece = "".join(scalar_text[start:end])
                token_id = self._pieces.get(piece)
                if token_id is None:
                    continue
                score = best[start] + float(self._scores[token_id])
                if score > best[end]:
                    best[end] = score
                    back_positions[end] = start
                    back_ids[end] = token_id

            unknown_score = best[end - 1] + self._unknown_penalty
            if unknown_score > best[end]:
                best[end] = unknown_score
                back_positions[end] = end - 1
                back_ids[end] = self.unk_id

        tokens: list[Token] = []
        position = length
        while position > 0:
            start = back_positions[position]
            tokens.append(Token(back_ids[position], tuple(scalar_text[start:position])))
            position = start
        tokens.reverse()
        return tokens
