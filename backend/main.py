from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Annotated, Any, List

import logging
import math
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    BaseSettings = BaseModel
    SettingsConfigDict = None

from src.capm import compute_beta_and_capm
from src.config import DEFAULT_END_DATE, DEFAULT_START_DATE, GLOBAL_BENCHMARK, TICKER_TO_NAME, get_local_benchmark
from src.indicators import compute_all_indicators
from src.markowitz import (
    efficient_frontier,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    simulate_portfolios,
    weights_table,
)
from src.portfolio_optimization import optimize_target_return
from src.preprocess import equal_weight_portfolio, equal_weight_vector
from src.returns_analysis import compute_return_series, descriptive_stats, normality_tests, stylized_facts_comment
from src.risk_metrics import risk_comparison_table, validate_weights
from src.services.macro_service import MacroService
from src.services.market_service import MarketService
from src.signals import evaluate_signals
from backend.cache import TTLCache


logger = logging.getLogger(__name__)

MACRO_CACHE_TTL_SECONDS = 3600
MARKET_CACHE_TTL_SECONDS = 3600
ANALYTICS_CACHE_TTL_SECONDS = 1800
"""
Estrategia de cache backend:
- macro snapshot: 3600s. Se invalida por TTL o reinicio del proceso.
- market/ticker bundle: 3600s. Se invalida por TTL, cambio de parametros o reinicio.
- bundles analiticos derivados de retornos: 1800s. Se invalida por TTL, cambio de parametros o reinicio.

La cache es local al proceso FastAPI; no sustituye el cache de Streamlit ni el
archivo persistente data/macro_cache.json.
"""


class BackendSettings(BaseSettings):
    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(env_file=".env", env_prefix="BACKEND_", extra="ignore")

    api_title: str = "RiskLab Backend"
    macro_cache_ttl_seconds: int = MACRO_CACHE_TTL_SECONDS
    market_cache_ttl_seconds: int = MARKET_CACHE_TTL_SECONDS
    analytics_cache_ttl_seconds: int = ANALYTICS_CACHE_TTL_SECONDS


@lru_cache(maxsize=1)
def build_settings() -> BackendSettings:
    return BackendSettings()


@lru_cache(maxsize=1)
def build_market_service() -> MarketService:
    return MarketService()


@lru_cache(maxsize=1)
def build_macro_service() -> MacroService:
    return MacroService()


_settings = build_settings()
app = FastAPI(title=_settings.api_title)
_market_service = build_market_service()
_macro_service = build_macro_service()
market_service = _market_service
macro_service = _macro_service
macro_snapshot_cache: TTLCache[MacroSnapshotResponse] = TTLCache(MACRO_CACHE_TTL_SECONDS, maxsize=8)
ticker_prices_cache: TTLCache[pd.DataFrame] = TTLCache(MARKET_CACHE_TTL_SECONDS, maxsize=256)
market_bundle_cache: TTLCache[dict[str, object]] = TTLCache(MARKET_CACHE_TTL_SECONDS, maxsize=128)
returns_bundle_cache: TTLCache[dict[str, object]] = TTLCache(ANALYTICS_CACHE_TTL_SECONDS, maxsize=128)

TickerStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DEFAULT_API_START_DATE = date.fromisoformat(DEFAULT_START_DATE)
DEFAULT_API_END_DATE = date.fromisoformat(DEFAULT_END_DATE)


class ValidationIssue(BaseModel):
    field: str = Field(..., examples=["tickers[1]"])
    message: str = Field(..., examples=["String should have at least 1 character"])


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["Solicitud inválida."])
    detail: List[ValidationIssue]


class ExternalProviderError(RuntimeError):
    pass


class TemporaryUnavailableError(RuntimeError):
    pass


JsonScalar = str | int | float | bool | None


class DynamicRecord(RootModel[dict[str, JsonScalar]]):
    pass


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    status: str
    service: str


class MacroSnapshotResponse(BaseModel):
    risk_free_rate_pct: float | None = None
    inflation_yoy: float | None = None
    cop_per_usd: float | None = None
    usdcop_market: float | None = None
    source: str | None = None
    last_updated: str | None = None


class OhlcvRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str | None = Field(default=None, alias="Date")
    open: float | None = Field(default=None, alias="Open")
    high: float | None = Field(default=None, alias="High")
    low: float | None = Field(default=None, alias="Low")
    close: float | None = Field(default=None, alias="Close")
    adj_close: float | None = Field(default=None, alias="Adj Close")
    volume: int | float | None = Field(default=None, alias="Volume")


class ReturnRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str | None = Field(default=None, alias="Date")
    simple_return: float | None = None
    log_return: float | None = None


class DescriptiveStatRecord(BaseModel):
    metric: str
    value: float | None


class NormalityTestRecord(BaseModel):
    test: str
    statistic: float | None
    p_value: float | None


class SignalDetails(BaseModel):
    macd_buy: bool
    macd_sell: bool
    rsi_buy: bool
    rsi_sell: bool
    boll_buy: bool
    boll_sell: bool
    golden_cross: bool
    death_cross: bool
    stoch_buy: bool
    stoch_sell: bool


class SignalResult(BaseModel):
    score_buy: int
    score_sell: int
    recommendation: str
    color: str
    reasons: List[str]
    details: SignalDetails


class VarCvarMetricRecord(BaseModel):
    method: str
    var_daily: float
    cvar_daily: float
    var_annualized: float
    cvar_annualized: float


class WeightRecord(BaseModel):
    asset: str
    weight: float | None


class PortfolioSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    return_: float | None = Field(default=None, alias="return")
    volatility: float | None = None
    sharpe: float | None = None
    weights: List[WeightRecord] = Field(default_factory=list)


class TargetPortfolio(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    return_: float | None = Field(default=None, alias="return")
    volatility: float | None = None
    weights: List[WeightRecord] = Field(default_factory=list)


class FrontierPoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    volatility: float | None
    return_: float | None = Field(default=None, alias="return")


class CapmMetrics(BaseModel):
    beta: float | None = None
    alpha_diaria: float | None = None
    r_value: float | None = None
    r_squared: float | None = None
    p_value_beta: float | None = None
    expected_return_capm_annual: float | None = None
    classification: str | None = None


class MarketBundleRequest(BaseModel):
    tickers: List[TickerStr] = Field(
        ...,
        min_length=1,
        description="Lista de tickers a consultar.",
        examples=[["3382.T", "ATD.TO", "ACWI"]],
    )
    start: date = Field(
        ...,
        description="Fecha inicial en formato YYYY-MM-DD.",
        examples=["2024-01-01"],
    )
    end: date = Field(
        ...,
        description="Fecha final en formato YYYY-MM-DD.",
        examples=["2024-12-31"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "tickers": ["AAPL", "MSFT", "^GSPC"],
                    "start": "2024-01-01",
                    "end": "2024-12-31",
                },
                {
                    "tickers": ["3382.T", "ATD.TO", "FEMSAUBD.MX", "ACWI"],
                    "start": "2024-01-01",
                    "end": "2024-12-31",
                },
            ]
        }
    )

    @model_validator(mode="after")
    def validate_date_range(self) -> "MarketBundleRequest":
        if self.end < self.start:
            raise ValueError("`end` no puede ser anterior a `start`.")
        return self


class MarketBundleResponse(BaseModel):
    ohlcv: dict[str, List[OhlcvRecord]]
    close: List[DynamicRecord]
    returns: List[DynamicRecord]
    missing_tickers: List[str] = Field(default_factory=list)
    last_available_date: str | None = None
    calendar_diagnostics: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ohlcv": {
                    "AAPL": [
                        {
                            "Date": "2024-01-02T00:00:00",
                            "Open": 187.15,
                            "High": 188.44,
                            "Low": 183.89,
                            "Close": 185.64,
                            "Adj Close": 184.94,
                            "Volume": 82488700,
                        }
                    ]
                },
                "close": [
                    {
                        "Date": "2024-01-02T00:00:00",
                        "AAPL": 184.94,
                        "MSFT": 368.51,
                        "^GSPC": 4742.83,
                    }
                ],
                "returns": [
                    {
                        "Date": "2024-01-03T00:00:00",
                        "AAPL": -0.0075,
                        "MSFT": -0.0014,
                        "^GSPC": -0.0080,
                    }
                ],
            }
        }
    )


class ReturnsResponse(BaseModel):
    ticker: str
    start: date
    end: date
    returns: List[ReturnRecord]
    descriptive_stats: List[DescriptiveStatRecord]
    normality_tests: List[NormalityTestRecord]
    comment: str


class IndicatorsResponse(BaseModel):
    ticker: str
    start: date
    end: date
    indicators: List[DynamicRecord]


class SignalEvaluateRequest(BaseModel):
    ticker: TickerStr
    start: date = Field(default=DEFAULT_API_START_DATE)
    end: date = Field(default=DEFAULT_API_END_DATE)
    rsi_overbought: float = Field(default=70, gt=0, lt=100)
    rsi_oversold: float = Field(default=30, gt=0, lt=100)
    stoch_overbought: float = Field(default=80, gt=0, lt=100)
    stoch_oversold: float = Field(default=20, gt=0, lt=100)

    @model_validator(mode="after")
    def validate_signal_request(self) -> "SignalEvaluateRequest":
        if self.end < self.start:
            raise ValueError("`end` no puede ser anterior a `start`.")
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("`rsi_oversold` debe ser menor que `rsi_overbought`.")
        if self.stoch_oversold >= self.stoch_overbought:
            raise ValueError("`stoch_oversold` debe ser menor que `stoch_overbought`.")
        return self


class SignalEvaluateResponse(BaseModel):
    ticker: str
    start: date
    end: date
    signal: SignalResult


class RiskVarCvarRequest(BaseModel):
    tickers: List[TickerStr] = Field(..., min_length=1)
    start: date = Field(default=DEFAULT_API_START_DATE)
    end: date = Field(default=DEFAULT_API_END_DATE)
    weights: List[float] | None = None
    alpha: float = Field(default=0.95, gt=0, lt=1)
    n_sim: int = Field(default=10000, ge=1000)

    @field_validator("weights", mode="before")
    @classmethod
    def validate_weights_schema(cls, weights):
        if weights is None:
            return None

        if not isinstance(weights, (list, tuple)):
            raise ValueError("`weights` debe ser una lista de números.")

        if len(weights) == 0:
            raise ValueError("`weights` no puede estar vacío.")

        clean_weights = []
        for index, weight in enumerate(weights):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"`weights[{index}]` debe ser numérico.")

            numeric_weight = float(weight)
            if not math.isfinite(numeric_weight):
                raise ValueError(f"`weights[{index}]` debe ser un número finito.")

            if numeric_weight < 0:
                raise ValueError(f"`weights[{index}]` no puede ser negativo.")

            clean_weights.append(numeric_weight)

        weights_sum = sum(clean_weights)
        if not math.isclose(weights_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"`weights` debe sumar 1.0; suma recibida: {weights_sum:.6f}."
            )

        return clean_weights

    @model_validator(mode="after")
    def validate_risk_request(self) -> "RiskVarCvarRequest":
        if self.end < self.start:
            raise ValueError("`end` no puede ser anterior a `start`.")
        if self.weights is not None and len(self.weights) != len(self.tickers):
            raise ValueError(
                "`weights` debe tener la misma cantidad de elementos que `tickers`; "
                f"se recibieron {len(self.weights)} pesos para {len(self.tickers)} tickers."
            )
        return self


class RiskVarCvarResponse(BaseModel):
    tickers: List[str]
    start: date
    end: date
    alpha: float
    weights: List[float]
    table: List[VarCvarMetricRecord]


class MarkowitzRequest(BaseModel):
    tickers: List[TickerStr] = Field(..., min_length=2)
    start: date = Field(default=DEFAULT_API_START_DATE)
    end: date = Field(default=DEFAULT_API_END_DATE)
    rf_annual: float = Field(default=0.03, gt=-1)
    n_portfolios: int = Field(default=10000, ge=1000)
    target_return: float | None = Field(default=None)

    @model_validator(mode="after")
    def validate_markowitz_request(self) -> "MarkowitzRequest":
        if self.end < self.start:
            raise ValueError("`end` no puede ser anterior a `start`.")
        return self


class MarkowitzResponse(BaseModel):
    tickers: List[str]
    start: date
    end: date
    rf_annual: float
    n_portfolios_generated: int
    minimum_variance: PortfolioSummary
    maximum_sharpe: PortfolioSummary
    efficient_frontier: List[FrontierPoint]
    target_portfolio: TargetPortfolio | None = None


class CapmResponse(BaseModel):
    ticker: str
    benchmark: str
    start: date
    end: date
    rf_annual: float
    metrics: CapmMetrics


def get_market_service() -> MarketService:
    return build_market_service()


def get_macro_service() -> MacroService:
    return build_macro_service()


def get_settings() -> BackendSettings:
    return build_settings()


MarketServiceDep = Annotated[MarketService, Depends(get_market_service)]
MacroServiceDep = Annotated[MacroService, Depends(get_macro_service)]
SettingsDep = Annotated[BackendSettings, Depends(get_settings)]


def build_error_response(error: str, detail: List[dict[str, str]]) -> dict[str, JsonScalar | List[dict[str, str]]]:
    return {"error": error, "detail": detail}


def error_detail(field: str, message: str) -> List[dict[str, str]]:
    return [{"field": field, "message": message}]


def raise_api_error(status_code: int, error: str, field: str, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail=build_error_response(error, error_detail(field, message)),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        loc = [str(part) for part in err.get("loc", []) if part != "body"]
        if not loc:
            field = "request"
        else:
            field_parts = []
            for part in loc:
                if part.isdigit() and field_parts:
                    field_parts[-1] = f"{field_parts[-1]}[{part}]"
                else:
                    field_parts.append(part)
            field = ".".join(field_parts)

        details.append(
            {
                "field": field,
                "message": err.get("msg", "Valor inválido."),
            }
        )

    return JSONResponse(
        status_code=422,
        content=build_error_response("Solicitud inválida.", details),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail and "detail" in exc.detail:
        content = exc.detail
    else:
        content = build_error_response(
            "Error de solicitud.",
            [{"field": "request", "message": str(exc.detail)}],
        )

    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(ExternalProviderError)
async def external_provider_exception_handler(request, exc: ExternalProviderError):
    return JSONResponse(
        status_code=502,
        content=build_error_response(
            "Fallo de proveedor externo.",
            error_detail("provider", str(exc)),
        ),
    )


@app.exception_handler(TemporaryUnavailableError)
async def temporary_unavailable_exception_handler(request, exc: TemporaryUnavailableError):
    return JSONResponse(
        status_code=503,
        content=build_error_response(
            "Servicio temporalmente no disponible.",
            error_detail("service", str(exc)),
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled backend error", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content=build_error_response(
            "Servicio temporalmente no disponible.",
            error_detail("service", "Ocurrió un error inesperado al procesar la solicitud."),
        ),
    )


def to_json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if value is pd.NaT:
        return None

    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def dataframe_to_json_records(df: pd.DataFrame) -> List[dict[str, JsonScalar]]:
    if df is None or df.empty:
        return []

    frame = df.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index.name = "Date"
    frame = frame.reset_index()
    frame.columns = [str(col) for col in frame.columns]

    records: List[dict[str, JsonScalar]] = []
    for row in frame.to_dict(orient="records"):
        clean_row = {str(key): to_json_safe_value(value) for key, value in row.items()}
        records.append(clean_row)

    return records


def descriptive_stat_records(df: pd.DataFrame) -> List[DescriptiveStatRecord]:
    records = []
    for row in dataframe_to_json_records(df):
        records.append(
            DescriptiveStatRecord(
                metric=str(row.get("index", "")),
                value=row.get("valor") if isinstance(row.get("valor"), (int, float)) else None,
            )
        )
    return records


def return_records(df: pd.DataFrame) -> List[ReturnRecord]:
    rows = []
    for row in dataframe_to_json_records(df):
        date_value = row.get("Date") or row.get("index")
        rows.append(
            ReturnRecord(
                date=str(date_value) if date_value is not None else None,
                simple_return=row.get("simple_return") if isinstance(row.get("simple_return"), (int, float)) else None,
                log_return=row.get("log_return") if isinstance(row.get("log_return"), (int, float)) else None,
            )
        )
    return rows


def normality_test_records(df: pd.DataFrame) -> List[NormalityTestRecord]:
    records = []
    for row in dataframe_to_json_records(df):
        records.append(
            NormalityTestRecord(
                test=str(row.get("test", "")),
                statistic=row.get("estadistico") if isinstance(row.get("estadistico"), (int, float)) else None,
                p_value=row.get("p_value") if isinstance(row.get("p_value"), (int, float)) else None,
            )
        )
    return records


def risk_metric_records(df: pd.DataFrame) -> List[VarCvarMetricRecord]:
    rows = []
    for row in dataframe_to_json_records(df):
        method = row.get("método") or row.get("mÃ©todo") or row.get("metodo") or ""
        rows.append(
            VarCvarMetricRecord(
                method=str(method),
                var_daily=float(row.get("VaR_diario") or 0.0),
                cvar_daily=float(row.get("CVaR_diario") or 0.0),
                var_annualized=float(row.get("VaR_anualizado") or 0.0),
                cvar_annualized=float(row.get("CVaR_anualizado") or 0.0),
            )
        )
    return rows


def weight_records(df: pd.DataFrame) -> List[WeightRecord]:
    rows = []
    for row in dataframe_to_json_records(df):
        asset = row.get("activo") or row.get("Activo") or ""
        weight = row.get("peso") if "peso" in row else row.get("Peso")
        rows.append(
            WeightRecord(
                asset=str(asset),
                weight=weight if isinstance(weight, (int, float)) else None,
            )
        )
    return rows


def frontier_points(df: pd.DataFrame) -> List[FrontierPoint]:
    points = []
    for row in dataframe_to_json_records(df):
        points.append(
            FrontierPoint(
                volatility=row.get("volatility") if isinstance(row.get("volatility"), (int, float)) else None,
                return_=row.get("return") if isinstance(row.get("return"), (int, float)) else None,
            )
        )
    return points


def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(value) for value in obj]

    if isinstance(obj, np.ndarray):
        return [to_json_safe(value) for value in obj.tolist()]

    if isinstance(obj, pd.Series):
        return to_json_safe(obj.to_dict())

    if isinstance(obj, pd.DataFrame):
        return dataframe_to_json_records(obj)

    return to_json_safe_value(obj)


def get_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or df.empty:
        raise_api_error(
            404,
            "Recurso no encontrado.",
            "ticker",
            f"No hay datos disponibles para '{ticker}'.",
        )

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        raise_api_error(
            422,
            "Datos de mercado incompletos.",
            "ticker",
            f"'{ticker}' no tiene columnas de cierre disponibles.",
        )

    series = df[close_col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    return pd.to_numeric(series, errors="coerce").dropna()


def load_ticker_prices(market: MarketService, ticker: str, start: date, end: date) -> pd.DataFrame:
    if end < start:
        raise_api_error(
            400,
            "Parámetros inválidos.",
            "end",
            "`end` no puede ser anterior a `start`.",
        )

    cache_key = ("ticker_prices", ticker, start.isoformat(), end.isoformat())
    cached = ticker_prices_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    try:
        prices = market.get_prices(ticker=ticker, start=start.isoformat(), end=end.isoformat())
        ticker_prices_cache.set(cache_key, prices.copy())
        return prices
    except RuntimeError as exc:
        raise ExternalProviderError(
            f"No se pudo obtener datos de mercado para '{ticker}' desde el proveedor externo."
        ) from exc
    except Exception as exc:
        raise TemporaryUnavailableError(
            f"No fue posible procesar temporalmente la consulta de mercado para '{ticker}'."
        ) from exc


def load_returns_bundle(
    market: MarketService,
    tickers: List[str],
    start: date,
    end: date,
    min_assets: int = 1,
) -> dict[str, object]:
    cache_key = ("returns_bundle", tuple(tickers), start.isoformat(), end.isoformat(), min_assets)
    cached = returns_bundle_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        bundle = market.load_market_bundle(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
        )
    except RuntimeError as exc:
        raise ExternalProviderError(
            "No se pudo obtener el bundle de mercado desde el proveedor externo."
        ) from exc
    except Exception as exc:
        raise TemporaryUnavailableError(
            "No fue posible procesar temporalmente el bundle de retornos."
        ) from exc

    missing = [
        {"field": f"tickers[{index}]", "message": f"No se encontraron datos para '{ticker}'."}
        for index, ticker in enumerate(tickers)
        if bundle["ohlcv"].get(ticker) is None or bundle["ohlcv"].get(ticker).empty
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=build_error_response("Recurso no encontrado.", missing),
        )

    returns = bundle["returns"].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty or returns.shape[1] < min_assets:
        raise_api_error(
            422,
            "Datos insuficientes para el cálculo solicitado.",
            "tickers",
            "No hay suficientes retornos alineados para calcular la métrica.",
        )

    bundle["returns"] = returns
    returns_bundle_cache.set(cache_key, bundle)
    return bundle


def load_market_bundle_cached(market: MarketService, payload: MarketBundleRequest) -> dict[str, object]:
    cache_key = (
        "market_bundle",
        tuple(payload.tickers),
        payload.start.isoformat(),
        payload.end.isoformat(),
    )
    cached = market_bundle_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        bundle = market.load_market_bundle(
            tickers=payload.tickers,
            start=payload.start.isoformat(),
            end=payload.end.isoformat(),
        )
    except RuntimeError as exc:
        raise ExternalProviderError(
            "No se pudo obtener el bundle de mercado desde el proveedor externo."
        ) from exc
    except Exception as exc:
        raise TemporaryUnavailableError(
            "No fue posible procesar temporalmente el bundle de mercado."
        ) from exc
    market_bundle_cache.set(cache_key, bundle)
    return bundle


def get_macro_snapshot_cached(macro: MacroService) -> MacroSnapshotResponse:
    cache_key = "macro_snapshot"
    cached = macro_snapshot_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        snapshot = {key: to_json_safe_value(value) for key, value in macro.get_macro_snapshot().items()}
    except RuntimeError as exc:
        raise ExternalProviderError(
            "No se pudo obtener el snapshot macroeconómico desde los proveedores externos."
        ) from exc
    except Exception as exc:
        raise TemporaryUnavailableError(
            "No fue posible procesar temporalmente el snapshot macroeconómico."
        ) from exc
    response = MacroSnapshotResponse(**snapshot)
    macro_snapshot_cache.set(cache_key, response)
    return response


async def load_ticker_prices_async(market: MarketService, ticker: str, start: date, end: date) -> pd.DataFrame:
    return await run_in_threadpool(load_ticker_prices, market, ticker, start, end)


async def load_returns_bundle_async(
    market: MarketService,
    tickers: List[str],
    start: date,
    end: date,
    min_assets: int = 1,
) -> dict[str, object]:
    return await run_in_threadpool(load_returns_bundle, market, tickers, start, end, min_assets)


async def load_market_bundle_cached_async(
    market: MarketService,
    payload: MarketBundleRequest,
) -> dict[str, object]:
    return await run_in_threadpool(load_market_bundle_cached, market, payload)


async def get_macro_snapshot_cached_async(macro: MacroService) -> MacroSnapshotResponse:
    return await run_in_threadpool(get_macro_snapshot_cached, macro)


def resolve_weights(weights: List[float] | None, n_assets: int) -> np.ndarray:
    if weights is None:
        return equal_weight_vector(n_assets)

    try:
        return validate_weights(np.asarray(weights, dtype=float), n_assets=n_assets)
    except ValueError as exc:
        raise_api_error(
            422,
            "Pesos de portafolio inválidos.",
            "weights",
            str(exc),
        )


def portfolio_summary(row: pd.Series) -> PortfolioSummary:
    if row is None or row.empty:
        return PortfolioSummary()

    return PortfolioSummary(
        return_=row.get("return") if isinstance(row.get("return"), (int, float)) else None,
        volatility=row.get("volatility") if isinstance(row.get("volatility"), (int, float)) else None,
        sharpe=row.get("sharpe") if isinstance(row.get("sharpe"), (int, float)) else None,
        weights=weight_records(weights_table(row)),
    )


@app.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(status="ok", service="api")


@app.head("/")
def root_head() -> Response:
    return Response(status_code=200)


@app.get("/macro/snapshot", response_model=MacroSnapshotResponse)
async def macro_snapshot(macro: MacroServiceDep) -> MacroSnapshotResponse:
    return await get_macro_snapshot_cached_async(macro)


@app.post(
    "/market/bundle",
    response_model=MarketBundleResponse,
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Error de validación del request.",
            "content": {
                "application/json": {
                    "examples": {
                        "tickers_vacio": {
                            "summary": "Tickers vacío",
                            "value": {
                                "error": "Solicitud inválida.",
                                "detail": [
                                    {
                                        "field": "tickers",
                                        "message": "List should have at least 1 item after validation, not 0",
                                    }
                                ],
                            },
                        },
                        "ticker_en_blanco": {
                            "summary": "Ticker en blanco",
                            "value": {
                                "error": "Solicitud inválida.",
                                "detail": [
                                    {
                                        "field": "tickers[1]",
                                        "message": "String should have at least 1 character",
                                    }
                                ],
                            },
                        },
                        "fecha_invalida": {
                            "summary": "Fecha inválida",
                            "value": {
                                "error": "Solicitud inválida.",
                                "detail": [
                                    {
                                        "field": "start",
                                        "message": "Input should be a valid date or datetime, input is too short",
                                    }
                                ],
                            },
                        },
                        "rango_invalido": {
                            "summary": "end anterior a start",
                            "value": {
                                "error": "Solicitud inválida.",
                                "detail": [
                                    {
                                        "field": "request",
                                        "message": "Value error, `end` no puede ser anterior a `start`.",
                                    }
                                ],
                            },
                        },
                    }
                }
            },
        },
        404: {
            "model": ErrorResponse,
            "description": "Ticker inválido o sin datos en el rango solicitado.",
            "content": {
                "application/json": {
                    "example": {
                        "error": "No se encontraron datos para uno o más tickers.",
                        "detail": [
                            {
                                "field": "tickers[0]",
                                "message": "No se pudo descargar datos para 'STRING' o no hubo precios en el rango solicitado.",
                            }
                        ],
                    }
                }
            },
        },
    },
)
async def market_bundle(
    payload: MarketBundleRequest,
    market: MarketServiceDep,
) -> MarketBundleResponse:
    bundle = await load_market_bundle_cached_async(market, payload)

    missing_tickers = []
    for ticker in payload.tickers:
        df = bundle["ohlcv"].get(ticker)
        if df is None or df.empty:
            missing_tickers.append(ticker)

    valid_frames = [
        df
        for df in bundle["ohlcv"].values()
        if df is not None and not df.empty
    ]
    last_available_date = None
    if valid_frames:
        last_available = max(df.index.max() for df in valid_frames)
        if pd.notna(last_available):
            last_available_date = pd.Timestamp(last_available).date().isoformat()

    if len(missing_tickers) == len(payload.tickers):
        raise HTTPException(
            status_code=404,
            detail=build_error_response(
                "Recurso no encontrado.",
                [
                    {
                        "field": "tickers",
                        "message": (
                            f"Sin datos para el rango {payload.start}–{payload.end}. "
                            f"Último día disponible: {last_available_date or 'no disponible'}."
                        ),
                    }
                ],
            ),
        )

    return MarketBundleResponse(
        ohlcv={
            ticker: dataframe_to_json_records(df)
            for ticker, df in bundle["ohlcv"].items()
        },
        close=dataframe_to_json_records(bundle["close"]),
        returns=dataframe_to_json_records(bundle["returns"]),
        missing_tickers=missing_tickers,
        last_available_date=last_available_date,
        calendar_diagnostics=bundle.get("calendar_diagnostics", {}),
    )


@app.get("/returns/{ticker}", response_model=ReturnsResponse)
async def returns_analysis(
    ticker: str,
    market: MarketServiceDep,
    start: date = Query(default=DEFAULT_API_START_DATE),
    end: date = Query(default=DEFAULT_API_END_DATE),
) -> ReturnsResponse:
    prices = await load_ticker_prices_async(market=market, ticker=ticker, start=start, end=end)
    close = get_close_series(prices, ticker)
    returns_df = compute_return_series(close)

    if returns_df.empty:
        raise_api_error(
            422,
            "Datos insuficientes para calcular rendimientos.",
            "ticker",
            f"No hay suficientes precios para '{ticker}'.",
        )

    simple_returns = returns_df["simple_return"]
    return ReturnsResponse(
        ticker=ticker,
        start=start,
        end=end,
        returns=return_records(returns_df),
        descriptive_stats=descriptive_stat_records(descriptive_stats(simple_returns)),
        normality_tests=normality_test_records(normality_tests(simple_returns)),
        comment=stylized_facts_comment(simple_returns),
    )


@app.get("/indicators/{ticker}", response_model=IndicatorsResponse)
async def indicators(
    ticker: str,
    market: MarketServiceDep,
    start: date = Query(default=DEFAULT_API_START_DATE),
    end: date = Query(default=DEFAULT_API_END_DATE),
    sma_window: int = Query(default=20, ge=2),
    ema_window: int = Query(default=20, ge=2),
    rsi_window: int = Query(default=14, ge=2),
    bb_window: int = Query(default=20, ge=2),
    bb_std: float = Query(default=2.0, gt=0),
    stoch_window: int = Query(default=14, ge=2),
) -> IndicatorsResponse:
    prices = await load_ticker_prices_async(market=market, ticker=ticker, start=start, end=end)
    indicators_df = compute_all_indicators(
        prices,
        sma_window=sma_window,
        ema_window=ema_window,
        rsi_window=rsi_window,
        bb_window=bb_window,
        bb_std=bb_std,
        stoch_window=stoch_window,
    )

    return IndicatorsResponse(
        ticker=ticker,
        start=start,
        end=end,
        indicators=dataframe_to_json_records(indicators_df),
    )


@app.post("/signals/evaluate", response_model=SignalEvaluateResponse)
async def signals_evaluate(
    payload: SignalEvaluateRequest,
    market: MarketServiceDep,
) -> SignalEvaluateResponse:
    prices = await load_ticker_prices_async(market=market, ticker=payload.ticker, start=payload.start, end=payload.end)
    indicators_df = compute_all_indicators(prices)
    signal = evaluate_signals(
        indicators_df,
        rsi_overbought=payload.rsi_overbought,
        rsi_oversold=payload.rsi_oversold,
        stoch_overbought=payload.stoch_overbought,
        stoch_oversold=payload.stoch_oversold,
    )

    if not signal:
        raise_api_error(
            422,
            "Datos insuficientes para evaluar señales.",
            "ticker",
            f"No hay suficientes indicadores válidos para '{payload.ticker}'.",
        )

    return SignalEvaluateResponse(
        ticker=payload.ticker,
        start=payload.start,
        end=payload.end,
        signal=to_json_safe(signal),
    )


@app.post("/risk/var-cvar", response_model=RiskVarCvarResponse)
async def risk_var_cvar(
    payload: RiskVarCvarRequest,
    market: MarketServiceDep,
) -> RiskVarCvarResponse:
    bundle = await load_returns_bundle_async(
        market=market,
        tickers=payload.tickers,
        start=payload.start,
        end=payload.end,
        min_assets=1,
    )
    returns = bundle["returns"]
    weights = resolve_weights(payload.weights, n_assets=returns.shape[1])
    portfolio_returns = returns @ weights
    table = risk_comparison_table(
        portfolio_returns=portfolio_returns,
        asset_returns_df=returns,
        weights=weights,
        alpha=payload.alpha,
        n_sim=payload.n_sim,
    )

    if table.empty:
        raise_api_error(
            422,
            "No fue posible calcular VaR/CVaR.",
            "tickers",
            "La muestra no cumple los requisitos mínimos del cálculo.",
        )

    return RiskVarCvarResponse(
        tickers=payload.tickers,
        start=payload.start,
        end=payload.end,
        alpha=payload.alpha,
        weights=[float(value) for value in weights],
        table=risk_metric_records(table),
    )


@app.post("/portfolio/markowitz", response_model=MarkowitzResponse)
async def portfolio_markowitz(
    payload: MarkowitzRequest,
    market: MarketServiceDep,
) -> MarkowitzResponse:
    bundle = await load_returns_bundle_async(
        market=market,
        tickers=payload.tickers,
        start=payload.start,
        end=payload.end,
        min_assets=2,
    )
    returns = bundle["returns"]
    sim_df = simulate_portfolios(
        returns,
        rf_annual=payload.rf_annual,
        n_portfolios=payload.n_portfolios,
    )

    if sim_df.empty:
        raise_api_error(
            422,
            "No fue posible simular portafolios.",
            "tickers",
            "No hay suficientes retornos válidos para Markowitz.",
        )

    min_var = minimum_variance_portfolio(sim_df)
    max_sharpe = maximum_sharpe_portfolio(sim_df)
    frontier = efficient_frontier(sim_df)
    target_portfolio = None

    if payload.target_return is not None:
        target = optimize_target_return(returns, payload.target_return)
        if target is not None:
            target_portfolio = TargetPortfolio(
                return_=target["return"] if isinstance(target["return"], (int, float)) else None,
                volatility=target["volatility"] if isinstance(target["volatility"], (int, float)) else None,
                weights=[
                    WeightRecord(asset=str(asset), weight=to_json_safe_value(weight))
                    for asset, weight in zip(returns.columns, target["weights"])
                ],
            )

    return MarkowitzResponse(
        tickers=payload.tickers,
        start=payload.start,
        end=payload.end,
        rf_annual=payload.rf_annual,
        n_portfolios_generated=int(len(sim_df)),
        minimum_variance=portfolio_summary(min_var),
        maximum_sharpe=portfolio_summary(max_sharpe),
        efficient_frontier=frontier_points(frontier),
        target_portfolio=target_portfolio,
    )


@app.get("/capm/{ticker}", response_model=CapmResponse)
async def capm(
    ticker: str,
    market: MarketServiceDep,
    macro: MacroServiceDep,
    start: date = Query(default=DEFAULT_API_START_DATE),
    end: date = Query(default=DEFAULT_API_END_DATE),
    benchmark: str | None = Query(default=None),
) -> CapmResponse:
    if benchmark is None:
        asset_name = TICKER_TO_NAME.get(ticker)
        benchmark = get_local_benchmark(asset_name) if asset_name else GLOBAL_BENCHMARK

    bundle = await load_returns_bundle_async(
        market=market,
        tickers=[ticker, benchmark],
        start=start,
        end=end,
        min_assets=2,
    )
    returns = bundle["returns"]
    macro_snapshot_data = (await get_macro_snapshot_cached_async(macro)).model_dump()
    rf_pct = macro_snapshot_data.get("risk_free_rate_pct", 3.0)
    rf_annual = float(rf_pct) / 100 if rf_pct == rf_pct else 0.03

    result = compute_beta_and_capm(
        asset_returns=returns[ticker],
        market_returns=returns[benchmark],
        rf_annual=rf_annual,
    )

    if not result:
        raise_api_error(
            422,
            "No fue posible calcular CAPM.",
            "ticker",
            "No hay suficientes retornos alineados para el cálculo.",
        )

    metrics = {
        key: value
        for key, value in result.items()
        if key not in {"regression_line", "scatter_data"}
    }

    return CapmResponse(
        ticker=ticker,
        benchmark=benchmark,
        start=start,
        end=end,
        rf_annual=rf_annual,
        metrics=CapmMetrics(**to_json_safe(metrics)),
    )
