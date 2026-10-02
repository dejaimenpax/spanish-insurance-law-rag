"""Run retrieval evaluations and write reproducible reports."""

import json
import subprocess
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from insurance_rag.domain.models import Chunk
from insurance_rag.evaluation.dataset import Category, Dataset, Question
from insurance_rag.evaluation.retrieval_metrics import mean, score_retrieval

SearchFn = Callable[[str], Sequence[Chunk]]


@dataclass
class ConfigResult:
    name: str
    overall: dict[str, float]
    by_category: dict[str, dict[str, float]]
    per_question: list[dict[str, Any]] = field(default_factory=list)


def evaluate_retrieval(
    dataset: Dataset, configs: dict[str, SearchFn], ks: Sequence[int] = (1, 3, 5, 10)
) -> list[ConfigResult]:
    questions = [q for q in dataset.questions if q.category is not Category.OUT_OF_SCOPE]
    results = []
    for name, search in configs.items():
        rows = [_score_question(q, search, ks) for q in questions]
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_category[row["category"]].append(row)
        results.append(
            ConfigResult(
                name=name,
                overall=_aggregate(rows, ks),
                by_category={cat: _aggregate(rs, ks) for cat, rs in sorted(by_category.items())},
                per_question=rows,
            )
        )
    return results


def _score_question(question: Question, search: SearchFn, ks: Sequence[int]) -> dict[str, Any]:
    ranked = list(search(question.question))
    scores = score_retrieval(ranked, question.gold, ks)
    return {
        "id": question.id,
        "category": question.category.value,
        "rr": scores.reciprocal_rank,
        **{f"recall@{k}": v for k, v in scores.recall_at.items()},
        "top": [f"{c.norm_short_name} {c.label}" for c in ranked[:3]],
    }


def _aggregate(rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> dict[str, float]:
    metrics = {f"recall@{k}": round(mean([r[f"recall@{k}"] for r in rows]), 3) for k in ks}
    metrics["mrr"] = round(mean([r["rr"] for r in rows]), 3)
    metrics["n"] = len(rows)
    return metrics


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_report(
    results: Sequence[ConfigResult], out_dir: Path, *, metadata: dict[str, Any], prefix: str
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{prefix}-{stamp}.json"
    payload = {
        "metadata": {**metadata, "git": git_revision(), "created_at": stamp},
        "results": [r.__dict__ for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
    return path


def markdown_table(results: Sequence[ConfigResult], ks: Sequence[int] = (1, 3, 5, 10)) -> str:
    header = "| Config | " + " | ".join(f"R@{k}" for k in ks) + " | MRR |"
    lines = [header, "|" + "---|" * (len(ks) + 2)]
    for r in results:
        cells = " | ".join(f"{r.overall[f'recall@{k}']:.3f}" for k in ks)
        lines.append(f"| {r.name} | {cells} | {r.overall['mrr']:.3f} |")
    return "\n".join(lines)
