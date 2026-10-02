"""Client for the BOE open-data API (consolidated legislation).

Docs: https://www.boe.es/datosabiertos/documentos/APIconsolidada.pdf
"""

import time
from datetime import date, datetime
from types import TracebackType
from typing import Any, Self

import httpx
import structlog
from pydantic import BaseModel

BASE_URL = "https://www.boe.es/datosabiertos/api/legislacion-consolidada/id"
USER_AGENT = "spanish-insurance-law-rag (+https://github.com/dejaimenpax/spanish-insurance-law-rag)"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

log = structlog.get_logger(__name__)


class BoeApiError(RuntimeError):
    pass


class IndexEntry(BaseModel):
    block_id: str
    title: str
    updated_at: date


class NormMetadata(BaseModel):
    norm_id: str
    title: str
    rank: str
    entry_into_force: date | None
    eli_url: str | None
    consolidated_html_url: str
    repealed: bool


def parse_boe_date(value: str) -> date:
    """Parse BOE dates, either ``YYYYMMDD`` or ``YYYYMMDDTHHMMSSZ``."""
    return datetime.strptime(value[:8], "%Y%m%d").date()


class BoeClient:
    """Thin synchronous client with retries and a polite delay between requests."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 4,
        min_interval: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            transport=transport,
        )
        self._max_retries = max_retries
        self._min_interval = min_interval
        self._last_request = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def fetch_text_xml(self, norm_id: str) -> bytes:
        """Full consolidated text, with every version of every block."""
        return self._get(f"/{norm_id}/texto", accept="application/xml").content

    def fetch_block_xml(self, norm_id: str, block_id: str) -> bytes:
        return self._get(f"/{norm_id}/texto/bloque/{block_id}", accept="application/xml").content

    def fetch_index(self, norm_id: str) -> list[IndexEntry]:
        payload = self._get_json(f"/{norm_id}/texto/indice")
        return parse_index(payload)

    def fetch_metadata(self, norm_id: str) -> NormMetadata:
        payload = self._get_json(f"/{norm_id}/metadatos")
        return parse_metadata(payload)

    def _get_json(self, path: str) -> dict[str, Any]:
        payload: dict[str, Any] = self._get(path, accept="application/json").json()
        status = str(payload.get("status", {}).get("code", ""))
        if status != "200":
            raise BoeApiError(f"BOE API returned status {status} for {path}")
        return payload

    def _get(self, path: str, *, accept: str) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._http.get(path, headers={"Accept": accept})
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise BoeApiError(f"GET {path} failed: {exc}") from exc
                log.warning("boe.retry", path=path, attempt=attempt, error=str(exc))
            else:
                if response.status_code not in _RETRYABLE_STATUS:
                    if response.is_error:
                        raise BoeApiError(f"GET {path} returned HTTP {response.status_code}")
                    return response
                if attempt == self._max_retries:
                    raise BoeApiError(f"GET {path} returned HTTP {response.status_code}")
                log.warning("boe.retry", path=path, attempt=attempt, status=response.status_code)
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()


def parse_index(payload: dict[str, Any]) -> list[IndexEntry]:
    data = payload["data"]
    blocks = data[0]["bloque"] if isinstance(data, list) else data["bloque"]
    return [
        IndexEntry(
            block_id=b["id"],
            title=b.get("titulo") or "",
            updated_at=parse_boe_date(b["fecha_actualizacion"]),
        )
        for b in blocks
    ]


def parse_metadata(payload: dict[str, Any]) -> NormMetadata:
    data = payload["data"]
    item: dict[str, Any] = data[0] if isinstance(data, list) else data

    def text(value: Any) -> str:
        return str(value.get("texto", "")) if isinstance(value, dict) else str(value or "")

    vigencia = item.get("fecha_vigencia")
    return NormMetadata(
        norm_id=item["identificador"],
        title=text(item.get("titulo")),
        rank=text(item.get("rango")),
        entry_into_force=parse_boe_date(vigencia) if vigencia else None,
        eli_url=item.get("url_eli"),
        consolidated_html_url=item.get("url_html_consolidada")
        or f"https://www.boe.es/buscar/act.php?id={item['identificador']}",
        repealed=item.get("estatus_derogacion") == "S",
    )
