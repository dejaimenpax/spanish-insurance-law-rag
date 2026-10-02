"""Evaluation dataset: questions with gold provisions and reference answers."""

import re
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from insurance_rag.corpus.catalog import Catalog
from insurance_rag.domain.models import Chunk, ProvisionKind

_KIND_ALIASES = {
    "art.": ProvisionKind.ARTICULO,
    "DA": ProvisionKind.DISPOSICION_ADICIONAL,
    "DT": ProvisionKind.DISPOSICION_TRANSITORIA,
    "DD": ProvisionKind.DISPOSICION_DEROGATORIA,
    "DF": ProvisionKind.DISPOSICION_FINAL,
    "anexo": ProvisionKind.ANEXO,
}
_REF = re.compile(r"^(?P<norm>.+?) (?P<kind>art\.|DA|DT|DD|DF|anexo) (?P<number>.+)$")


class Category(StrEnum):
    SINGLE = "single"
    LITERAL = "literal"
    CROSS = "cross"
    OUT_OF_SCOPE = "out_of_scope"


class ProvisionKey(BaseModel):
    """Identifies a provision: by kind and number, or by block id for annexes."""

    model_config = ConfigDict(frozen=True)

    norm_id: str
    kind: ProvisionKind
    number: str | None = None
    block_id: str | None = None

    def matches(self, chunk: Chunk) -> bool:
        if chunk.norm_id != self.norm_id or chunk.kind is not self.kind:
            return False
        if self.block_id is not None:
            return chunk.block_id == self.block_id
        return chunk.number == self.number

    def label(self, catalog: Catalog) -> str:
        short = catalog.get(self.norm_id).short_name
        alias = next(a for a, k in _KIND_ALIASES.items() if k is self.kind)
        return f"{short} {alias} {self.block_id or self.number}"


class Question(BaseModel):
    id: str
    category: Category
    question: str
    gold: tuple[ProvisionKey, ...]
    answer: str


class Dataset(BaseModel):
    version: int
    review_status: str
    questions: tuple[Question, ...]


def parse_ref(ref: str, catalog: Catalog) -> ProvisionKey:
    match = _REF.match(ref.strip())
    if match is None:
        raise ValueError(f"Malformed gold reference: {ref!r}")
    by_short = {n.short_name: n.id for n in catalog.norms}
    if match["norm"] not in by_short:
        raise ValueError(f"Unknown norm in gold reference: {ref!r}")
    kind = _KIND_ALIASES[match["kind"]]
    norm_id = by_short[match["norm"]]
    if kind is ProvisionKind.ANEXO:
        return ProvisionKey(norm_id=norm_id, kind=kind, block_id=match["number"])
    return ProvisionKey(norm_id=norm_id, kind=kind, number=match["number"])


def load_dataset(path: Path, catalog: Catalog) -> Dataset:
    raw = yaml.safe_load(path.read_text("utf-8"))
    questions = []
    for item in raw["questions"]:
        gold = tuple(parse_ref(ref, catalog) for ref in item.get("gold") or ())
        questions.append(Question.model_validate({**item, "gold": gold}))
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate question ids: {sorted(duplicates)}")
    for q in questions:
        if (q.category is Category.OUT_OF_SCOPE) != (not q.gold):
            raise ValueError(f"{q.id}: only out_of_scope questions may have no gold provisions")
    return Dataset(
        version=raw["version"], review_status=raw["review_status"], questions=tuple(questions)
    )
