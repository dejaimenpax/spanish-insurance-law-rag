"""Core domain model: norms, their provisions and the chunks that get indexed.

The model separates national and EU law, and directly applicable acts from directives that need
transposition, and keeps entry into force apart from date of application. Phase 1 only ingests
Spanish law, but answers can already warn when they cite something that does not apply yet.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Jurisdiction(StrEnum):
    ES = "ES"
    EU = "EU"


class NormRank(StrEnum):
    LEY = "ley"
    REAL_DECRETO_LEY = "real_decreto_ley"
    REAL_DECRETO_LEGISLATIVO = "real_decreto_legislativo"
    REAL_DECRETO = "real_decreto"
    DIRECTIVA = "directiva"
    REGLAMENTO_UE = "reglamento_ue"
    REGLAMENTO_DELEGADO_UE = "reglamento_delegado_ue"


class LegalEffect(StrEnum):
    NATIONAL = "national"
    DIRECTLY_APPLICABLE = "directly_applicable"
    REQUIRES_TRANSPOSITION = "requires_transposition"


class ProvisionKind(StrEnum):
    ARTICULO = "articulo"
    DISPOSICION_ADICIONAL = "disposicion_adicional"
    DISPOSICION_TRANSITORIA = "disposicion_transitoria"
    DISPOSICION_DEROGATORIA = "disposicion_derogatoria"
    DISPOSICION_FINAL = "disposicion_final"
    ANEXO = "anexo"


class ParagraphKind(StrEnum):
    TEXT = "text"
    HEADING = "heading"
    """Heading inside a block, e.g. the title of an annex section or of a table."""
    TABLE = "table"
    NOTE = "note"
    """Editorial note added by the BOE to the consolidated text (e.g. updated amounts)."""


class Scope(BaseModel):
    """Which blocks of a norm are indexed, when only part of it is in scope.

    Blocks are selected by their BOE block id: every block from ``from_block`` up to, but not
    including, ``until_block``, plus each block listed in ``extra_blocks``.
    """

    model_config = ConfigDict(frozen=True)

    from_block: str
    until_block: str
    extra_blocks: tuple[str, ...] = ()


class NormSpec(BaseModel):
    """A norm as declared in the corpus catalog."""

    model_config = ConfigDict(frozen=True)

    id: str
    short_name: str
    title: str
    aliases: tuple[str, ...] = ()
    rank: NormRank
    jurisdiction: Jurisdiction
    legal_effect: LegalEffect
    topics: tuple[str, ...]
    scope: Scope | None = None
    scope_rationale: str | None = None
    application_date: date | None = None
    """Date from which the norm applies, when it differs from entry into force."""


class Norm(BaseModel):
    """A catalog norm enriched with source metadata fetched at ingestion time."""

    spec: NormSpec
    entry_into_force: date | None
    consolidated_as_of: date
    """Date of the latest update reflected in the consolidated text."""
    source_url: str
    eli_url: str | None = None
    reuse_terms: str


class Paragraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ParagraphKind
    text: str
    css_class: str | None = None


class Provision(BaseModel):
    """One article, additional/transitional/repealing/final provision or annex."""

    norm_id: str
    block_id: str
    kind: ProvisionKind
    label: str
    """Title as written in the text, e.g. "Artículo dieciocho"."""
    number: str | None
    """Canonical number, e.g. "18", "6 bis", "13" or "única"."""
    heading: str | None
    """Rubric that follows the label, e.g. "Grandes riesgos"."""
    hierarchy: tuple[str, ...]
    """Enclosing headings from the outermost inwards, e.g. ("TÍTULO I. …", "CAPÍTULO II. …")."""
    paragraphs: tuple[Paragraph, ...]
    version_in_force_since: date
    block_updated_at: date
    repealed: bool = False
    amendment_notes: tuple[str, ...] = ()


class Chunk(BaseModel):
    """The unit that is embedded, indexed, retrieved and cited."""

    chunk_id: str
    norm_id: str
    norm_short_name: str
    block_id: str
    seq: int = 0
    """Position of the chunk within its provision."""
    kind: ProvisionKind
    label: str
    number: str | None
    heading: str | None
    apartado: str | None
    hierarchy: tuple[str, ...]
    text: str
    embed_text: str
    url: str
    version_in_force_since: date
    block_updated_at: date
    consolidated_as_of: date
    repealed: bool = False
    jurisdiction: Jurisdiction
    legal_effect: LegalEffect
    rank: NormRank
    topics: tuple[str, ...]
    application_date: date | None = None
    refs_out: tuple[str, ...] = Field(default=())
    """Chunk-independent references to other provisions, as "<norm_id>#<number>" keys."""

    @property
    def citation_label(self) -> str:
        """Human-readable citation, e.g. "LCS, art. 18.1"."""
        if self.kind is ProvisionKind.ARTICULO:
            base = f"art. {self.number or self.label}"
        else:
            base = self.label
        if self.apartado:
            base = (
                f"{base}.{self.apartado}"
                if self.kind is ProvisionKind.ARTICULO
                else (f"{base}, apartado {self.apartado}")
            )
        return f"{self.norm_short_name}, {base}"
