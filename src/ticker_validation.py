from __future__ import annotations

from pydantic import BaseModel, field_validator

from src.config import ASSETS


PORTFOLIO_TICKERS = {meta["ticker"].upper(): meta["ticker"] for meta in ASSETS.values()}
PORTFOLIO_TICKER_TO_ASSET = {meta["ticker"].upper(): name for name, meta in ASSETS.items()}
PORTFOLIO_ASSET_LIST_TEXT = "Seven & i Holdings, Alimentation Couche-Tard, FEMSA, BP y Carrefour"
PORTFOLIO_VALIDATION_MESSAGE = (
    "Por ahora no tenemos habilitado ese activo en el portafolio. "
    f"Actualmente puedes consultar: {PORTFOLIO_ASSET_LIST_TEXT}."
)


class TickerInput(BaseModel):
    ticker: str

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return str(value or "").strip().upper()

    @field_validator("ticker")
    @classmethod
    def ticker_must_belong_to_portfolio(cls, value: str) -> str:
        if value not in PORTFOLIO_TICKERS:
            raise ValueError(PORTFOLIO_VALIDATION_MESSAGE)
        return PORTFOLIO_TICKERS[value]


def validate_portfolio_ticker(ticker: str) -> TickerInput:
    return TickerInput(ticker=ticker)


def asset_name_for_ticker(ticker: str) -> str | None:
    return PORTFOLIO_TICKER_TO_ASSET.get(str(ticker or "").strip().upper())
