from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000"
BACKEND_TIMEOUT_SECONDS = 30


class BackendAPIError(RuntimeError):
    def __init__(
        self,
        user_message: str,
        technical_detail: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.technical_detail = technical_detail
        self.status_code = status_code


def backend_base_url() -> str:
    return os.getenv("BACKEND_API_BASE_URL", DEFAULT_BACKEND_BASE_URL).rstrip("/")


def friendly_error_message(exc: Exception, default: str = "Ocurrió un error inesperado al consultar la API del proyecto.") -> str:
    if isinstance(exc, BackendAPIError):
        return exc.user_message
    return default


def _extract_backend_message(response: requests.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None

    detail = payload.get("detail")
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict) and first.get("message"):
            return str(first["message"])

    if payload.get("error"):
        return str(payload["error"])

    return None


def _message_for_http_status(status_code: int, response: requests.Response) -> str:
    backend_detail = _extract_backend_message(response)

    if status_code == 400:
        return "Los parámetros enviados no son válidos. Revisa fechas o configuración."
    if status_code == 404:
        return "No hay datos disponibles para el activo o rango seleccionado."
    if status_code == 422:
        return backend_detail or "Los datos enviados no son válidos. Revisa la configuración seleccionada."
    if status_code == 502:
        return "La API del proyecto no pudo obtener datos del proveedor externo."
    if status_code == 503:
        return "La API está temporalmente no disponible. Intenta nuevamente."

    return backend_detail or "La API del proyecto devolvió un error inesperado."


def _request_backend(method: str, path: str, **kwargs) -> dict[str, Any]:
    url = f"{backend_base_url()}{path}"
    try:
        response = requests.request(method, url, timeout=BACKEND_TIMEOUT_SECONDS, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as exc:
        raise BackendAPIError(
            f"No se pudo conectar con la API del proyecto. Verifica que FastAPI esté corriendo en {backend_base_url()}.",
            technical_detail=str(exc),
        ) from exc
    except requests.Timeout as exc:
        raise BackendAPIError(
            "La API del proyecto tardó demasiado en responder. Intenta nuevamente.",
            technical_detail=str(exc),
        ) from exc
    except requests.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else None
        raise BackendAPIError(
            _message_for_http_status(status_code or 0, response) if response is not None else "La API del proyecto devolvió un error inesperado.",
            technical_detail=str(exc),
            status_code=status_code,
        ) from exc
    except requests.RequestException as exc:
        raise BackendAPIError(
            "No fue posible completar la consulta a la API del proyecto.",
            technical_detail=str(exc),
        ) from exc


def backend_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return _request_backend("GET", path, params=params)


def backend_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _request_backend("POST", path, json=payload)


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    return df
