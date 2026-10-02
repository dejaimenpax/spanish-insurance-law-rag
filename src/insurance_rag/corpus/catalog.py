"""Load the corpus catalog shipped with the package."""

from functools import lru_cache
from importlib import resources

import yaml
from pydantic import BaseModel

from insurance_rag.domain.models import NormSpec


class Catalog(BaseModel):
    norms: tuple[NormSpec, ...]

    def get(self, norm_id: str) -> NormSpec:
        for norm in self.norms:
            if norm.id == norm_id:
                return norm
        raise KeyError(norm_id)


@lru_cache
def load_catalog() -> Catalog:
    raw = resources.files("insurance_rag.corpus").joinpath("catalog.yaml").read_text("utf-8")
    return Catalog.model_validate(yaml.safe_load(raw))
