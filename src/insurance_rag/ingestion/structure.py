"""Turn the flat list of BOE blocks into provisions that know where they sit in the norm."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from xml.etree.ElementTree import Element

import structlog

from insurance_rag.domain.models import (
    NormSpec,
    Paragraph,
    ParagraphKind,
    Provision,
    ProvisionKind,
)
from insurance_rag.ingestion.boe.parser import RawBlock, element_text, parse_version
from insurance_rag.ingestion.spanish_numbers import normalize, parse_provision_number

log = structlog.get_logger(__name__)

# Heading levels, outermost first. Normalized (lowercase, no accents) title prefixes.
_HEADING_LEVELS: tuple[tuple[str, int], ...] = (
    ("libro", 0),
    ("titulo", 1),
    ("capitulo", 2),
    ("subseccion", 4),
    ("seccion", 3),
)
_PROVISION_KINDS: tuple[tuple[str, ProvisionKind], ...] = (
    ("articulo", ProvisionKind.ARTICULO),
    ("disposicion adicional", ProvisionKind.DISPOSICION_ADICIONAL),
    ("disposicion transitoria", ProvisionKind.DISPOSICION_TRANSITORIA),
    ("disposicion derogatoria", ProvisionKind.DISPOSICION_DEROGATORIA),
    ("disposicion final", ProvisionKind.DISPOSICION_FINAL),
    ("anexo", ProvisionKind.ANEXO),
    ("anejo", ProvisionKind.ANEXO),
)
# Heading that opens the text approved by a "por el que se aprueba ..." instrument.
_APPROVED_TEXT_MARKERS = ("texto", "reglamento")
_SKIPPED_BLOCK_TYPES = frozenset({"preambulo", "firma", "nota_inicial"})
_REPEALED = re.compile(
    r"^\(?\s*(derogad[oa]s?|sin contenido|suprimid[oa]s?|anulad[oa]s?|queda(n)? derogad[oa]s?)\b",
    re.IGNORECASE,
)


@dataclass
class _HierarchyStack:
    entries: list[tuple[int, str]] = field(default_factory=list)

    def push(self, level: int, heading: str) -> None:
        self.entries = [(lvl, h) for lvl, h in self.entries if lvl < level]
        self.entries.append((level, heading))

    def clear(self) -> None:
        self.entries = []

    def snapshot(self) -> tuple[str, ...]:
        return tuple(heading for _, heading in self.entries)


def heading_level(title: str) -> int | None:
    norm = normalize(title)
    for prefix, level in _HEADING_LEVELS:
        if norm.startswith(prefix):
            return level
    return None


def provision_kind(title: str) -> ProvisionKind | None:
    norm = normalize(title)
    for prefix, kind in _PROVISION_KINDS:
        if norm.startswith(prefix):
            return kind
    return None


def approved_text_start(blocks: Sequence[RawBlock]) -> int:
    """Index of the first block of the approved text, or 0 if the norm has no such split."""
    for i, block in enumerate(blocks):
        if block.block_type == "encabezado" and normalize(block.title) in _APPROVED_TEXT_MARKERS:
            return i + 1
    return 0


def select_in_scope(spec: NormSpec, blocks: Sequence[RawBlock]) -> list[RawBlock]:
    """Blocks that belong to the indexed part of the norm, in document order."""
    candidates = list(blocks[approved_text_start(blocks) :])
    if spec.scope is None:
        return candidates
    ids = [b.block_id for b in candidates]
    for required in (spec.scope.from_block, spec.scope.until_block, *spec.scope.extra_blocks):
        if required not in ids:
            raise ValueError(f"{spec.id}: scope block {required!r} not found in the text")
    start = ids.index(spec.scope.from_block)
    end = ids.index(spec.scope.until_block)
    extra = set(spec.scope.extra_blocks)
    return [b for i, b in enumerate(candidates) if start <= i < end or b.block_id in extra]


def build_provisions(
    spec: NormSpec,
    blocks: Sequence[RawBlock],
    *,
    block_updates: Mapping[str, date],
    as_of: date,
) -> list[Provision]:
    """Build the provisions in force on ``as_of`` for the in-scope part of a norm."""
    stack = _HierarchyStack()
    provisions: list[Provision] = []

    for block in select_in_scope(spec, blocks):
        if block.block_type in _SKIPPED_BLOCK_TYPES:
            continue
        version = block.version_as_of(as_of)
        if version is None:
            log.debug("structure.not_in_force", norm=spec.id, block=block.block_id)
            continue

        kind = provision_kind(block.title)
        level = heading_level(block.title)
        is_heading = block.block_type in ("encabezado", "cabecera") or (
            kind is None and level is not None
        )
        if is_heading and kind is not ProvisionKind.ANEXO:
            if level is None:
                log.debug("structure.unknown_heading", norm=spec.id, block=block.block_id)
                continue
            stack.push(level, _heading_text(block.title, version.element))
            continue

        if kind is None:
            # Untitled or unusual content blocks (e.g. standalone tables of an annex).
            kind = ProvisionKind.ANEXO
            log.debug("structure.untyped_block", norm=spec.id, block=block.block_id)
        if kind is not ProvisionKind.ARTICULO:
            stack.clear()

        parsed = parse_version(version.element)
        paragraphs = parsed.paragraphs
        label = block.title or (parsed.label_line or block.block_id)
        heading = _rubric(label, parsed.label_line)
        if kind is ProvisionKind.ANEXO:
            heading, paragraphs = _annex_heading(label, paragraphs)

        provisions.append(
            Provision(
                norm_id=spec.id,
                block_id=block.block_id,
                kind=kind,
                label=label,
                number=parse_provision_number(label) if kind is not ProvisionKind.ANEXO else None,
                heading=heading,
                hierarchy=stack.snapshot(),
                paragraphs=paragraphs,
                version_in_force_since=version.in_force_since,
                block_updated_at=block_updates.get(block.block_id, version.in_force_since),
                repealed=_is_repealed(paragraphs, parsed.label_line),
                amendment_notes=parsed.amendment_notes,
            )
        )
    return provisions


def _heading_text(title: str, element: Element) -> str:
    """Full heading, e.g. "CAPÍTULO I. Objeto y ámbito de aplicación"."""
    lines = [element_text(p) for p in element if p.tag == "p" and element_text(p)]
    if not lines:
        return title
    head, *rest = lines
    return f"{head}. {' '.join(rest)}".rstrip(". ") if rest else head


def _rubric(label: str, label_line: str | None) -> str | None:
    """Rubric that follows the label in "Artículo 11. Grandes riesgos."."""
    if not label_line:
        return None
    norm_label, norm_line = normalize(label), normalize(label_line)
    if not norm_line.startswith(norm_label):
        return None
    rest = label_line[len(label) :].strip(" .:-\u2013")
    return rest or None


def _annex_heading(
    label: str, paragraphs: tuple[Paragraph, ...]
) -> tuple[str | None, tuple[Paragraph, ...]]:
    """Annex blocks repeat their label as heading lines; use the first other heading as rubric."""
    remaining = list(paragraphs)
    heading: str | None = None
    while remaining and remaining[0].kind is ParagraphKind.HEADING:
        text = remaining.pop(0).text
        if normalize(text) == normalize(label) or normalize(text) in ("anexo", "anejo"):
            continue
        heading = text
        break
    return heading, tuple(remaining)


def _is_repealed(paragraphs: Sequence[Paragraph], label_line: str | None) -> bool:
    body = [p.text for p in paragraphs if p.kind is ParagraphKind.TEXT]
    if label_line and not body:
        # "Artículo 6 bis. (Derogado)." keeps the status in the label line itself.
        return bool(re.search(r"\(\s*(derogad|sin contenido|suprimid|anulad)", label_line, re.I))
    return len(body) == 1 and bool(_REPEALED.match(body[0])) and len(body[0]) < 200
