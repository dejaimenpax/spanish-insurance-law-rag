"""Parse the XML consolidated text returned by the BOE API.

The response is a flat sequence of ``<bloque>`` elements (headings, provisions, preamble,
signature…). Each block holds one ``<version>`` per wording it has had; the wording in force
is the one with the latest ``fecha_vigencia`` not after the reference date.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from xml.etree.ElementTree import Element

from defusedxml import ElementTree

from insurance_rag.domain.models import Paragraph, ParagraphKind
from insurance_rag.ingestion.boe.client import parse_boe_date

AMENDMENT_NOTE_CLASSES = frozenset({"nota_pie", "nota_pie_2"})
LABEL_CLASSES = frozenset({"articulo"})
HEADING_CLASSES = frozenset(
    {
        "anexo_num",
        "anexo_tit",
        "anexo",
        "libro",
        "libro_num",
        "libro_tit",
        "titulo",
        "titulo_num",
        "titulo_tit",
        "capitulo",
        "capitulo_num",
        "capitulo_tit",
        "seccion",
        "subseccion",
        "centro_negrita",
        "centro_cursiva",
        "centro_redonda",
    }
)
SKIPPED_CLASSES = frozenset({"imagen", "firma_rey", "firma_ministro"})

# Tables at least this numeric and this large are replaced by a descriptive stub: their values
# are actuarial coefficients that are useless to embed and better read at the source.
NUMERIC_TABLE_RATIO = 0.8
NUMERIC_TABLE_MIN_CELLS = 24
MAX_STUB_CAPTIONS = 8

_NUMERIC_CELL = re.compile(r"^[\s\d.,%+\-\u2013()/€*]*$")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class RawVersion:
    in_force_since: date
    element: Element


@dataclass(frozen=True)
class RawBlock:
    block_id: str
    block_type: str
    title: str
    versions: tuple[RawVersion, ...]
    expired_on: date | None = None
    """Set when the whole block was repealed or lapsed (``fecha_caducidad``)."""

    def version_as_of(self, as_of: date) -> RawVersion | None:
        """The wording in force on ``as_of``, or None if the block was not in force then."""
        if self.expired_on is not None and self.expired_on <= as_of:
            return None
        candidates = [v for v in self.versions if v.in_force_since <= as_of]
        return max(candidates, key=lambda v: v.in_force_since) if candidates else None


@dataclass(frozen=True)
class ParsedVersion:
    label_line: str | None
    paragraphs: tuple[Paragraph, ...]
    amendment_notes: tuple[str, ...]


def clean_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\xa0", " ")).strip()


def element_text(element: Element) -> str:
    return clean_text("".join(element.itertext()))


def parse_blocks(xml: bytes) -> list[RawBlock]:
    root = ElementTree.fromstring(xml)
    status = root.findtext("status/code")
    if status not in (None, "200"):
        raise ValueError(f"BOE response has status {status}")
    blocks: list[RawBlock] = []
    for block in root.iter("bloque"):
        versions = tuple(RawVersion(_version_date(v), v) for v in block.findall("version"))
        blocks.append(
            RawBlock(
                block_id=block.get("id", ""),
                block_type=block.get("tipo", ""),
                title=clean_text(block.get("titulo", "")),
                versions=versions,
                expired_on=parse_boe_date(expiry)
                if (expiry := block.get("fecha_caducidad"))
                else None,
            )
        )
    return blocks


def _version_date(version: Element) -> date:
    return parse_boe_date(version.get("fecha_vigencia") or version.get("fecha_publicacion") or "")


def parse_version(version: Element) -> ParsedVersion:
    """Split a block version into its label line, content paragraphs and amendment notes."""
    label_line: str | None = None
    paragraphs: list[Paragraph] = []
    notes: list[str] = []
    pending_captions: list[str] = []

    for child in version:
        css = child.get("class")
        if child.tag == "table":
            table = _render_table(child, pending_captions)
            pending_captions = []
            _append_table(paragraphs, table)
            continue

        text = element_text(child)
        if not text or css in SKIPPED_CLASSES:
            continue
        if child.tag == "blockquote":
            paragraphs.extend(_flush_captions(pending_captions))
            pending_captions = []
            paragraphs.append(Paragraph(kind=ParagraphKind.NOTE, text=text, css_class=css))
        elif css in AMENDMENT_NOTE_CLASSES:
            notes.append(text)
        elif css in LABEL_CLASSES and label_line is None and not paragraphs:
            label_line = text
        elif css in HEADING_CLASSES and css.startswith("centro"):
            # Centered lines usually introduce the next table; keep them until we know.
            pending_captions.append(text)
        elif css in HEADING_CLASSES:
            paragraphs.extend(_flush_captions(pending_captions))
            pending_captions = []
            paragraphs.append(Paragraph(kind=ParagraphKind.HEADING, text=text, css_class=css))
        else:
            paragraphs.extend(_flush_captions(pending_captions))
            pending_captions = []
            paragraphs.append(Paragraph(kind=ParagraphKind.TEXT, text=text, css_class=css))

    paragraphs.extend(_flush_captions(pending_captions))
    return ParsedVersion(label_line, tuple(paragraphs), tuple(notes))


def _flush_captions(captions: list[str]) -> Iterator[Paragraph]:
    for caption in captions:
        yield Paragraph(kind=ParagraphKind.HEADING, text=caption, css_class="centro")


@dataclass
class _Table:
    caption: str
    rows: list[list[str]]
    header_rows: int
    numeric_ratio: float
    body_cells: int

    @property
    def is_numeric(self) -> bool:
        return (
            self.numeric_ratio >= NUMERIC_TABLE_RATIO and self.body_cells >= NUMERIC_TABLE_MIN_CELLS
        )


def _render_table(table: Element, pending_captions: list[str]) -> _Table:
    caption_el = table.find("caption")
    captions = [*pending_captions]
    if caption_el is not None and element_text(caption_el):
        captions.append(element_text(caption_el))
    rows: list[list[str]] = []
    header_rows = 0
    for tr in table.iter("tr"):
        cells = [element_text(c) for c in tr if c.tag in ("td", "th")]
        if not any(cells):
            continue
        if all(c.tag == "th" for c in tr if c.tag in ("td", "th")) and len(rows) == header_rows:
            header_rows += 1
        rows.append(cells)
    body = [cell for row in rows[header_rows:] for cell in row if cell]
    numeric = sum(1 for cell in body if _NUMERIC_CELL.match(cell))
    return _Table(
        caption=" ".join(captions),
        rows=rows,
        header_rows=header_rows,
        numeric_ratio=numeric / len(body) if body else 0.0,
        body_cells=len(body),
    )


def _append_table(paragraphs: list[Paragraph], table: _Table) -> None:
    if not table.rows:
        return
    header = " | ".join(cell for cell in table.rows[0] if cell) if table.header_rows else ""
    data_rows = len(table.rows) - table.header_rows
    previous = paragraphs[-1] if paragraphs else None
    if (
        previous is not None
        and previous.css_class == "numeric_table"
        and table.numeric_ratio >= NUMERIC_TABLE_RATIO
        and not _starts_new_table(previous.text, table.caption)
    ):
        # Continuations of a long numeric table, however small, collapse into a single stub.
        merged = _merge_stub(previous.text, table.caption, data_rows)
        paragraphs[-1] = Paragraph(kind=ParagraphKind.TABLE, text=merged, css_class="numeric_table")
        return
    if not table.is_numeric:
        paragraphs.append(
            Paragraph(kind=ParagraphKind.TABLE, text=_markdown(table), css_class="table")
        )
        return
    text = _stub_text([table.caption] if table.caption else [], header, data_rows)
    paragraphs.append(Paragraph(kind=ParagraphKind.TABLE, text=text, css_class="numeric_table"))


_TABLE_CODE = re.compile(
    r"\btabla\s+(?:t[eé]cnica\s+)?.*?\(?(\d+(?:\.[0-9A-Za-z]+)+|TT\d+)\)?", re.I
)


def _table_code(caption: str) -> str | None:
    match = _TABLE_CODE.search(caption)
    return match.group(1).upper() if match else None


def _starts_new_table(stub: str, caption: str) -> bool:
    """A caption naming a different table code ("Tabla 1.C.2") starts a new stub."""
    new_code = _table_code(caption)
    if new_code is None:
        return False
    match = _STUB_RE.match(stub)
    return match is None or _table_code(match["captions"]) != new_code


_STUB_RE = re.compile(
    r"^\[Tabla numérica\. Títulos: (?P<captions>.*?)\. Columnas: (?P<header>.*?)\. "
    r"(?P<rows>\d+) filas\. "
)


def _stub_text(captions: list[str], header: str, rows: int) -> str:
    shown = captions[:MAX_STUB_CAPTIONS]
    caption_text = "; ".join(shown) + ("; …" if len(captions) > MAX_STUB_CAPTIONS else "")
    return (
        f"[Tabla numérica. Títulos: {caption_text or 'sin título'}. "
        f"Columnas: {header or 'sin cabecera'}. {rows} filas. "
        "Los valores no se reproducen aquí; consúltense en la fuente oficial.]"
    )


def _merge_stub(stub: str, caption: str, rows: int) -> str:
    match = _STUB_RE.match(stub)
    if match is None:
        return stub
    captions = [c for c in match["captions"].split("; ") if c not in ("sin título", "…")]
    if caption and "continuaci" not in caption.lower() and caption not in captions:
        captions.append(caption)
    header = "" if match["header"] == "sin cabecera" else match["header"]
    return _stub_text(captions, header, int(match["rows"]) + rows)


def _markdown(table: _Table) -> str:
    width = max(len(row) for row in table.rows)
    rows = [
        [cell.replace("|", "/") for cell in row] + [""] * (width - len(row)) for row in table.rows
    ]
    header_count = max(table.header_rows, 1)
    lines = [f"| {' | '.join(row)} |" for row in rows[:header_count]]
    lines.append("|" + " --- |" * width)
    lines.extend(f"| {' | '.join(row)} |" for row in rows[header_count:])
    if table.caption:
        lines.insert(0, table.caption)
    return "\n".join(lines)
