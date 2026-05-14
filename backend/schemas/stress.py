"""Schemas for stress testing model requests and responses."""

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PortfolioStressRequest(BaseModel):
    """Input data for portfolio stress testing by asset shocks."""

    returns: list[list[float]] = Field(..., min_length=1)
    weights: list[float] = Field(..., min_length=1)
    shocks: dict[str, float] = Field(..., min_length=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_portfolio_shape(self) -> "PortfolioStressRequest":
        column_count = len(self.returns[0]) if self.returns else 0
        if column_count == 0:
            raise ValueError("returns debe contener al menos una columna de activos.")

        for row in self.returns:
            if len(row) != column_count:
                raise ValueError("Cada fila de returns debe tener la misma cantidad de activos.")
            if any(not math.isfinite(value) for value in row):
                raise ValueError("returns debe contener solo valores finitos.")

        if len(self.weights) != column_count:
            raise ValueError("weights debe tener el mismo tamano que las columnas de returns.")
        if any(not math.isfinite(value) for value in self.weights):
            raise ValueError("weights debe contener solo valores finitos.")
        if not math.isclose(sum(self.weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("weights debe sumar aproximadamente 1.")

        valid_shock_keys = {f"asset_{index}" for index in range(column_count)}
        unknown_keys = set(self.shocks) - valid_shock_keys
        if unknown_keys:
            raise ValueError("shocks solo puede usar llaves asset_0, asset_1, etc.")
        if any(not math.isfinite(value) for value in self.shocks.values()):
            raise ValueError("shocks debe contener solo valores finitos.")

        return self


class PortfolioStressResponse(BaseModel):
    """Summary statistics for a stressed portfolio scenario."""

    mean_return: float
    min_return: float
    max_drawdown: float
    var: float
    cvar: float
    observations: int


class CombinedStressScenarioRequest(BaseModel):
    """Input data for a combined adverse stress scenario."""

    prices: list[float] | None = None
    returns: list[float] | None = None
    weights: float | None = 1.0
    price_shock: float | None = None
    volatility_multiplier: float | None = None
    bond_price: float | None = None
    modified_duration: float | None = None
    convexity: float | None = None
    rate_shock: float | None = None

    @field_validator(
        "weights",
        "price_shock",
        "volatility_multiplier",
        "bond_price",
        "modified_duration",
        "convexity",
        "rate_shock",
    )
    @classmethod
    def validate_optional_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Los valores numericos deben ser finitos.")
        return value

    @field_validator("prices", "returns")
    @classmethod
    def validate_optional_series(cls, values: list[float] | None) -> list[float] | None:
        if values is not None and any(not math.isfinite(value) for value in values):
            raise ValueError("Las series deben contener solo valores finitos.")
        return values

    @model_validator(mode="after")
    def validate_scenario_components(self) -> "CombinedStressScenarioRequest":
        has_price_component = self.prices is not None and self.price_shock is not None
        has_volatility_component = self.returns is not None and self.volatility_multiplier is not None
        has_bond_component = any(
            value is not None
            for value in (self.bond_price, self.modified_duration, self.rate_shock)
        )

        if not (has_price_component or has_volatility_component or has_bond_component):
            raise ValueError("Debe venir al menos un componente del escenario.")

        bond_values = (self.bond_price, self.modified_duration, self.rate_shock)
        if any(value is not None for value in bond_values) and any(value is None for value in bond_values):
            raise ValueError("bond_price, modified_duration y rate_shock deben venir juntos.")

        if self.bond_price is not None and self.bond_price <= 0:
            raise ValueError("bond_price debe ser mayor que 0.")
        if self.modified_duration is not None and self.modified_duration < 0:
            raise ValueError("modified_duration debe ser mayor o igual que 0.")
        if self.convexity is not None and self.convexity < 0:
            raise ValueError("convexity debe ser mayor o igual que 0.")
        if self.volatility_multiplier is not None and self.volatility_multiplier <= 0:
            raise ValueError("volatility_multiplier debe ser mayor que 0.")

        return self


class CombinedStressScenarioResponse(BaseModel):
    """Result payload for a combined stress scenario."""

    scenario_summary: dict[str, Any] | None = None
    shocked_bond_price: float | None = None
    shocked_prices: list[float] | None = None
    shocked_returns: list[float] | None = None

