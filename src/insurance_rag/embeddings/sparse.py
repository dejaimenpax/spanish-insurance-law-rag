"""BM25 as sparse vectors for Qdrant.

Documents are encoded with the BM25 term-frequency saturation; Qdrant multiplies by IDF at query
time (``Modifier.IDF`` on the sparse vector), so the collection behaves like a BM25 index without
another service. Tokenization keeps what matters for legal references: numbers such as "10",
"20/2015" or "1.º" survive, and Latin suffixes ("bis") are kept as tokens.
"""

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass

import snowballstemmer

from insurance_rag.ingestion.spanish_numbers import normalize

# Function words that carry no retrieval signal in Spanish legal text.
SPANISH_STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquello
    aquellos asi aun aunque cada como con contra cual cuales cualquier cuando de del desde donde
    dos el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estas este esto
    estos fue fueron ha han hasta hay la las le les lo los mas me mi mientras muy ni no nos o os
    otra otras otro otros para pero por porque que quien quienes se sea sean segun ser si sin
    sino sobre su sus tal tambien tan tanto te tiene tienen todo todos tu un una unas uno unos y
    ya
    """.split()  # noqa: SIM905 - easier to maintain as a block of words
)
_TOKEN = re.compile(r"\d+(?:[./]\d+)*|[a-zñ]+")


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class Bm25Encoder:
    def __init__(self, *, k1: float = 1.2, b: float = 0.75, avg_doc_len: float = 250.0) -> None:
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len
        self._stemmer = snowballstemmer.stemmer("spanish")

    def tokenize(self, text: str) -> list[str]:
        tokens = []
        for raw in _TOKEN.findall(normalize(text)):
            if raw in SPANISH_STOPWORDS:
                continue
            tokens.append(raw if raw[0].isdigit() else self._stemmer.stemWord(raw))
        return tokens

    def fit_avg_doc_len(self, texts: list[str]) -> float:
        """Set the average document length used for length normalisation."""
        if texts:
            self.avg_doc_len = sum(len(self.tokenize(t)) for t in texts) / len(texts)
        return self.avg_doc_len

    def encode_document(self, text: str) -> SparseVector:
        tokens = self.tokenize(text)
        counts = Counter(tokens)
        norm = 1 - self.b + self.b * len(tokens) / self.avg_doc_len
        weights: dict[int, float] = {}
        for term, tf in counts.items():
            index = token_id(term)
            weights[index] = weights.get(index, 0.0) + tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return _to_vector(weights)

    def encode_query(self, text: str) -> SparseVector:
        return _to_vector({token_id(term): 1.0 for term in set(self.tokenize(text))})


def token_id(token: str) -> int:
    """Stable 31-bit id for a token (Python's hash() is salted per process)."""
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def _to_vector(weights: dict[int, float]) -> SparseVector:
    items = sorted((i, w) for i, w in weights.items() if not math.isclose(w, 0.0))
    return SparseVector(indices=[i for i, _ in items], values=[w for _, w in items])
