"""Split provisions into retrieval chunks that keep their legal coordinates.

Strategy:

* An article that fits in ``max_chars`` is a single chunk, so short articles keep their context.
* Longer articles are split at numbered paragraphs ("apartados": ``1.``, ``2.º``…), which is
  how they are cited. A marker only counts as a new apartado when it continues the sequence,
  so inner enumerations ("1.º …" inside apartado 3) do not break it.
* Anything still too long (an apartado, an annex section, a table) is packed into windows of
  whole paragraphs; long tables are split by rows repeating the header.

Every chunk carries the hierarchy (book, title, chapter…) in ``embed_text`` so that lexical and
dense retrieval both see where the text sits in the norm.
"""

import re
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from insurance_rag.domain.models import Chunk, Norm, Paragraph, ParagraphKind, Provision

DEFAULT_MAX_CHARS = 3200  # roughly 800 tokens of Spanish legal text
_CHUNK_NAMESPACE = uuid.UUID("5b0e1f0c-5a8e-4c38-9a53-2a1f5c3e7d10")
_APARTADO = re.compile(r"^(?P<num>\d{1,3})\s?(?:\.\s?º|\.º|º|\.\s?ª|ª|\.)\s+\S")
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


@dataclass(frozen=True)
class _Piece:
    apartado: str | None
    paragraphs: tuple[Paragraph, ...]


def boe_url(norm_id: str, block_id: str) -> str:
    return f"https://www.boe.es/buscar/act.php?id={norm_id}#{block_id}"


def chunk_id(norm_id: str, block_id: str, seq: int) -> str:
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{norm_id}#{block_id}#{seq}"))


def chunk_provision(
    norm: Norm, provision: Provision, *, max_chars: int = DEFAULT_MAX_CHARS
) -> list[Chunk]:
    pieces = list(_split(provision.paragraphs, max_chars))
    if not pieces:
        pieces = [_Piece(None, ())]
    spec = norm.spec
    chunks: list[Chunk] = []
    for seq, piece in enumerate(pieces):
        text = render(piece.paragraphs) or _empty_text(provision)
        chunks.append(
            Chunk(
                chunk_id=chunk_id(spec.id, provision.block_id, seq),
                norm_id=spec.id,
                norm_short_name=spec.short_name,
                block_id=provision.block_id,
                seq=seq,
                kind=provision.kind,
                label=provision.label,
                number=provision.number,
                heading=provision.heading,
                apartado=piece.apartado,
                hierarchy=provision.hierarchy,
                text=text,
                embed_text=embed_text(norm, provision, piece.apartado, text),
                url=boe_url(spec.id, provision.block_id),
                version_in_force_since=provision.version_in_force_since,
                block_updated_at=provision.block_updated_at,
                consolidated_as_of=norm.consolidated_as_of,
                repealed=provision.repealed,
                jurisdiction=spec.jurisdiction,
                legal_effect=spec.legal_effect,
                rank=spec.rank,
                topics=spec.topics,
                application_date=spec.application_date,
            )
        )
    return chunks


def render(paragraphs: Sequence[Paragraph]) -> str:
    lines = []
    for p in paragraphs:
        lines.append(
            f"[Nota de la edición consolidada: {p.text}]"
            if p.kind is ParagraphKind.NOTE
            else p.text
        )
    return "\n".join(lines)


def embed_text(norm: Norm, provision: Provision, apartado: str | None, text: str) -> str:
    title = provision.label if not provision.heading else f"{provision.label}. {provision.heading}"
    if apartado:
        title = f"{title} (apartado {apartado})"
    header = [f"{norm.spec.short_name} — {norm.spec.title}", *provision.hierarchy, title]
    return "\n".join([*header, text])


def _empty_text(provision: Provision) -> str:
    return "(Sin contenido)" if provision.repealed else provision.label


def _split(paragraphs: Sequence[Paragraph], max_chars: int) -> Iterator[_Piece]:
    if _size(paragraphs) <= max_chars:
        yield _Piece(None, tuple(paragraphs))
        return
    for apartado, group in _group_by_apartado(paragraphs):
        if _size(group) <= max_chars:
            yield _Piece(apartado, tuple(group))
        else:
            for window in _windows(group, max_chars):
                yield _Piece(apartado, window)


def _group_by_apartado(
    paragraphs: Sequence[Paragraph],
) -> Iterator[tuple[str | None, list[Paragraph]]]:
    current: str | None = None
    group: list[Paragraph] = []
    expected = 1
    for p in paragraphs:
        match = _APARTADO.match(p.text) if p.kind is ParagraphKind.TEXT else None
        if match and int(match["num"]) == expected:
            if group and current is not None:
                yield current, group
                group = []
            # Text before apartado 1 (an introduction) stays with apartado 1.
            current = match["num"]
            expected += 1
        group.append(p)
    if group:
        yield current, group


def _windows(paragraphs: Sequence[Paragraph], max_chars: int) -> Iterator[tuple[Paragraph, ...]]:
    window: list[Paragraph] = []
    for p in _explode(paragraphs, max_chars):
        starts_section = p.kind is ParagraphKind.HEADING and bool(window)
        if window and (starts_section or _size([*window, p]) > max_chars):
            # Do not leave a dangling heading at the end of a window.
            carry = [window.pop()] if window[-1].kind is ParagraphKind.HEADING else []
            if window:
                yield tuple(window)
            window = carry
        window.append(p)
    if window:
        yield tuple(window)


def _explode(paragraphs: Sequence[Paragraph], max_chars: int) -> Iterator[Paragraph]:
    """Break single paragraphs that exceed the limit: tables by rows, text by sentences."""
    for p in paragraphs:
        if len(p.text) <= max_chars:
            yield p
        elif p.kind is ParagraphKind.TABLE and "| --- |" in p.text:
            yield from _split_table(p, max_chars)
        else:
            yield from _split_sentences(p, max_chars)


def _split_table(p: Paragraph, max_chars: int) -> Iterator[Paragraph]:
    lines = p.text.split("\n")
    sep = next(i for i, line in enumerate(lines) if line.startswith("|") and "---" in line)
    head = lines[: sep + 1]
    body: list[str] = []
    for line in lines[sep + 1 :]:
        if body and len("\n".join([*head, *body, line])) > max_chars:
            yield Paragraph(kind=p.kind, text="\n".join([*head, *body]), css_class=p.css_class)
            body = []
        body.append(line)
    if body:
        yield Paragraph(kind=p.kind, text="\n".join([*head, *body]), css_class=p.css_class)


def _split_sentences(p: Paragraph, max_chars: int) -> Iterator[Paragraph]:
    buffer = ""
    for sentence in _SENTENCE_END.split(p.text):
        if buffer and len(buffer) + len(sentence) + 1 > max_chars:
            yield Paragraph(kind=p.kind, text=buffer, css_class=p.css_class)
            buffer = ""
        buffer = f"{buffer} {sentence}".strip()
    if buffer:
        yield Paragraph(kind=p.kind, text=buffer, css_class=p.css_class)


def _size(paragraphs: Sequence[Paragraph]) -> int:
    return sum(len(p.text) + 1 for p in paragraphs)
