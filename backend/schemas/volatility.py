"""Schemas for volatility model requests and responses."""

import math

from pydantic import BaseModel, Field, field_validator


class EWMAVolatilityRequest(BaseModel):
    """Input data for EWMA volatility calculations."""

    returns: list[float] = Field(..., min_length=2)
    lambda_: float = Field(default=0.94, gt=0, lt=1)
    annualize: bool = Field(default=False)
    periods_per_year: int = Field(default=252, gt=0)

    @field_validator("returns")
    @classmethod
    def validate_returns(cls, returns: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in returns):
            raise ValueError("returns debe contener solo valores finitos.")
        return returns


class EWMAVolatilityResponse(BaseModel):
    """EWMA volatility calculation result."""

    ewma_volatility: float
    ewma_variance: float | None = None
    lambda_: float
    annualize: bool
    periods_per_year: int
    observations: int

