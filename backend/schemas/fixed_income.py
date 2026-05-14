"""Schemas for fixed income model requests and responses."""

import math

from pydantic import BaseModel, Field, field_validator, model_validator


class BondMetricsRequest(BaseModel):
    """Input data for bond price, duration, and convexity metrics."""

    face_value: float = Field(..., gt=0)
    coupon_rate: float = Field(..., ge=0)
    market_rate: float = Field(..., ge=0)
    maturity_years: float = Field(..., gt=0)
    frequency: int = Field(default=1)

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, frequency: int) -> int:
        if frequency not in {1, 2, 4, 12}:
            raise ValueError("frequency debe ser uno de 1, 2, 4 o 12.")
        return frequency

    @model_validator(mode="after")
    def validate_periods(self) -> "BondMetricsRequest":
        periods = self.maturity_years * self.frequency
        if not math.isclose(periods, round(periods), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("maturity_years * frequency debe producir un numero entero de periodos.")
        return self


class BondMetricsResponse(BaseModel):
    """Bond valuation and risk metrics."""

    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float


class NelsonSiegelRequest(BaseModel):
    """Input data for Nelson-Siegel yield curve calculations."""

    maturities: list[float] = Field(..., min_length=1)
    beta0: float
    beta1: float
    beta2: float
    tau: float = Field(..., gt=0)

    @field_validator("maturities")
    @classmethod
    def validate_maturities(cls, maturities: list[float]) -> list[float]:
        if any(value <= 0 or not math.isfinite(value) for value in maturities):
            raise ValueError("maturities debe contener valores positivos y finitos.")
        return maturities

    @field_validator("beta0", "beta1", "beta2", "tau")
    @classmethod
    def validate_finite_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Todos los valores deben ser finitos.")
        return value


class NelsonSiegelResponse(BaseModel):
    """Nelson-Siegel yields for requested maturities."""

    maturities: list[float]
    yields: list[float]

