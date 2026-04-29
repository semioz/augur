import json
from pathlib import Path

import regex

# pre-tokenization split: common contractions, letter runs, number runs,
# punctuation runs, whitespace. Applied before bpe so merges can't cross chunks.
_PAT = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def _build_byte_encoder() -> dict[int, str]:
    # All 256 byte values must round-trip uniquely as printable unicode chars.
    # Printable ASCII + latin-1 ranges map to themselves; the remaining ~70 bytes
    # (control chars, whitespace) get pushed into the unused U+0100..U+0142 range.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _build_byte_encoder()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

class Tokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self._cache: dict[str, str] = {}

    @classmethod
    def from_files(cls, vocab_path: Path, merges_path: Path) -> "Tokenizer":
        vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        merges: list[tuple[str, str]] = []
        for line in merges_path.read_text(encoding="utf-8").splitlines()[1:]:
            if line:
                a, b = line.split()
                merges.append((a, b))
        return cls(vocab, merges)

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            return token
        while True:
            # Pick the highest-priority merge (lowest rank) currently in the word.
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self._cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in regex.findall(_PAT, text):
            unicode_chunk = "".join(_BYTE_ENCODER[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(unicode_chunk).split(" "):
                ids.append(self.encoder[piece])
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[i] for i in ids)
        return bytearray(_BYTE_DECODER[c] for c in text).decode("utf-8", errors="replace")