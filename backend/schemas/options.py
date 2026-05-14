"""Schemas for option pricing model requests and responses."""

import math

from pydantic import BaseModel, Field, field_validator


class BlackScholesRequest(BaseModel):
    """Input data for Black-Scholes option pricing and Greeks."""

    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    time_to_maturity: float = Field(..., gt=0)

    @field_validator("rate")
    @classmethod
    def validate_rate(cls, rate: float) -> float:
        if not math.isfinite(rate):
            raise ValueError("rate debe ser finito.")
        return rate


class BlackScholesResponse(BaseModel):
    """Black-Scholes call and put prices."""

    call_price: float
    put_price: float


class GreeksResponse(BaseModel):
    """Option Greeks for call and put contracts."""

    delta_call: float
    delta_put: float
    gamma: float
    vega: float
    theta_call: float
    theta_put: float
    rho_call: float
    rho_put: float

