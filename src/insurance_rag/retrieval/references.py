"""Detect literal references to provisions.

Examples: "artículo 10 de la Ley 50/1980", "art. 20.3 LCS", "disposición adicional decimotercera
del RDL 3/2020".

Used at query time to look provisions up exactly, and at ingestion time to record the
cross-references a provision makes.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from insurance_rag.domain.models import NormSpec, ProvisionKind
from insurance_rag.ingestion.spanish_numbers import normalize, parse_number

_NUM = r"(?P<num>\d+|[a-z]+(?:\s+y\s+[a-z]+)?)"
_SUFFIX = r"(?:\s*(?P<suffix>bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?"
_APARTADO = r"(?:\s*[.,]?\s*(?:apartado\s+)?(?P<apartado>\d+)(?:\s*[.º°ª])?)?"
_KIND_PATTERNS: tuple[tuple[ProvisionKind, str], ...] = (
    (ProvisionKind.ARTICULO, r"\b(?:articulos?|arts?\.?)\s*"),
    (
        ProvisionKind.DISPOSICION_ADICIONAL,
        r"\b(?:disposicion(?:es)?\s+adicional(?:es)?|d\.?\s?a\.?)\s*",
    ),
    (
        ProvisionKind.DISPOSICION_TRANSITORIA,
        r"\b(?:disposicion(?:es)?\s+transitoria(?:s)?|d\.?\s?t\.?)\s*",
    ),
    (ProvisionKind.DISPOSICION_FINAL, r"\b(?:disposicion(?:es)?\s+final(?:es)?|d\.?\s?f\.?)\s*"),
)
_ARTICLE_RE = re.compile(
    _KIND_PATTERNS[0][1] + r"(?P<num>\d+)" + _SUFFIX + r"(?P<rest>(?:\s*\.\s*\d+)?)"
)
_ORDINAL_PROVISION_RE = [
    (kind, re.compile(pattern + r"(?P<num>\d+|[a-z]+)(?:\s*[.º°ª])?"))
    for kind, pattern in _KIND_PATTERNS[1:]
]
# "de la Ley 20/2015", "del RD 1060/2015", "de esta Ley"…
_NORM_WINDOW = 90


@dataclass(frozen=True)
class ProvisionRef:
    kind: ProvisionKind
    number: str
    apartado: str | None
    norm_id: str | None
    """None when the text does not say which norm (or says "esta Ley")."""


class ReferenceParser:
    def __init__(self, norms: Sequence[NormSpec]) -> None:
        aliases: list[tuple[str, str]] = []
        for spec in norms:
            for alias in (spec.short_name, *spec.aliases):
                aliases.append((normalize(alias), spec.id))
        # Longest aliases first so "Real Decreto 1060/2015" beats "Real Decreto".
        aliases.sort(key=lambda item: len(item[0]), reverse=True)
        self._aliases = aliases

    def resolve_norm(self, text: str) -> str | None:
        """First norm mentioned in ``text`` (already normalized), if any."""
        best: tuple[int, str] | None = None
        for alias, norm_id in self._aliases:
            match = re.search(rf"(?<![\w/]){re.escape(alias)}(?![\w/])", text)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), norm_id)
        return best[1] if best else None

    def mentioned_norms(self, text: str) -> list[str]:
        norm_text = normalize(text)
        found = []
        for alias, norm_id in self._aliases:
            if norm_id not in found and re.search(
                rf"(?<![\w/]){re.escape(alias)}(?![\w/])", norm_text
            ):
                found.append(norm_id)
        return found

    def parse(self, text: str) -> list[ProvisionRef]:
        norm_text = normalize(text)
        refs: list[ProvisionRef] = []
        for match in _ARTICLE_RE.finditer(norm_text):
            number = match["num"] + (f" {match['suffix']}" if match["suffix"] else "")
            apartado = re.sub(r"\D", "", match["rest"]) or None
            refs.append(
                ProvisionRef(
                    ProvisionKind.ARTICULO,
                    number,
                    apartado,
                    self._norm_after(norm_text, match.end()),
                )
            )
        for kind, pattern in _ORDINAL_PROVISION_RE:
            for match in pattern.finditer(norm_text):
                value = parse_number(match["num"])
                if value is None:
                    continue
                refs.append(
                    ProvisionRef(kind, str(value), None, self._norm_after(norm_text, match.end()))
                )
        return list(dict.fromkeys(refs))

    def _norm_after(self, text: str, position: int) -> str | None:
        window = text[position : position + _NORM_WINDOW]
        # Stop at the next provision mention: "art. 3 y art. 5 del RD" is handled per mention.
        cut = re.search(r"\b(articulo|art\.|disposicion)\b", window)
        if cut:
            window = window[: cut.start()]
        if not re.match(r"^[\s,.]*(de|del|de la|de los)\b", window):
            return None
        return self.resolve_norm(window)
