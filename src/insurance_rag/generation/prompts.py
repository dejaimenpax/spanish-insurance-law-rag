"""Prompt construction: system instructions and retrieved provisions as citable search results."""

from collections.abc import Sequence
from datetime import date
from typing import Any

from insurance_rag.domain.models import Chunk

ABSTENTION = "No consta en la normativa indexada."

SYSTEM_PROMPT = f"""\
Eres un asistente que responde preguntas sobre la normativa de seguros aplicable en España \
usando exclusivamente los fragmentos normativos que se te proporcionan como resultados de búsqueda.

Reglas:
1. Responde en español, de forma clara y concisa, para un lector no especialista.
2. Basa cada afirmación en los fragmentos proporcionados y cítalos. No uses conocimiento propio \
sobre el contenido de las normas: si algo no está en los fragmentos, no lo afirmes.
3. Identifica siempre la norma, el artículo (o disposición) y, cuando el texto lo permita, el \
apartado, por ejemplo "artículo 20.4 de la Ley 50/1980, de Contrato de Seguro".
4. Si los fragmentos no contienen la respuesta, responde exactamente: "{ABSTENTION}" y, como \
mucho, una frase indicando qué normas se han consultado. No respondas con información general.
5. Si un fragmento está marcado como "AÚN NO APLICABLE", dilo expresamente al citarlo e indica la \
fecha desde la que se aplica.
6. Si un fragmento está marcado como derogado o sin contenido, no lo presentes como vigente.
7. No des asesoramiento jurídico ni recomendaciones personales: explica lo que dice la norma.
"""


def result_title(chunk: Chunk, *, today: date) -> str:
    title = chunk.citation_label
    if chunk.heading:
        title += f" — {chunk.heading}"
    notes = [f"texto consolidado a {chunk.consolidated_as_of.isoformat()}"]
    if chunk.application_date and chunk.application_date > today:
        notes.append(f"AÚN NO APLICABLE: se aplica desde {chunk.application_date.isoformat()}")
    if chunk.repealed:
        notes.append("DEROGADO o sin contenido")
    return f"{title} ({'; '.join(notes)})"


def search_result_block(chunk: Chunk, *, today: date) -> dict[str, Any]:
    """One retrieved chunk as a citable search result.

    Each paragraph is its own text block so that citations point at the paragraph used.
    """
    header = " › ".join((chunk.norm_short_name, *chunk.hierarchy, chunk.label))
    paragraphs = [p for p in chunk.text.split("\n") if p.strip()] or [chunk.text]
    return {
        "type": "search_result",
        "source": chunk.url,
        "title": result_title(chunk, today=today),
        "content": [{"type": "text", "text": header}]
        + [{"type": "text", "text": p} for p in paragraphs],
        "citations": {"enabled": True},
    }


def user_content(question: str, chunks: Sequence[Chunk], *, today: date) -> list[dict[str, Any]]:
    blocks = [search_result_block(c, today=today) for c in chunks]
    blocks.append(
        {
            "type": "text",
            "text": f"Fecha de hoy: {today.isoformat()}.\n\nPregunta: {question}",
        }
    )
    return blocks
