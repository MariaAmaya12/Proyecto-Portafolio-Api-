"""FastAPI routes for stress testing models."""

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.schemas.stress import (
    CombinedStressScenarioRequest,
    CombinedStressScenarioResponse,
    PortfolioStressRequest,
    PortfolioStressResponse,
)
from src.stress_testing import (
    combined_adverse_scenario,
    stress_portfolio_returns,
    summarize_stress_scenario,
)


router = APIRouter(
    prefix="/stress",
    tags=["stress"],
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return [float(item) for item in value.to_numpy(dtype=float)]
    if isinstance(value, pd.DataFrame):
        return {
            str(column): [float(item) for item in value[column].to_numpy(dtype=float)]
            for column in value.columns
        }
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.astype(float).ravel()]
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


@router.post("/portfolio", response_model=PortfolioStressResponse)
def calculate_portfolio_stress(payload: PortfolioStressRequest) -> PortfolioStressResponse:
    """Calculate stressed portfolio returns and summary risk metrics."""
    try:
        columns = [f"asset_{index}" for index in range(len(payload.returns[0]))]
        returns = pd.DataFrame(payload.returns, columns=columns)
        portfolio_returns = stress_portfolio_returns(
            returns,
            weights=payload.weights,
            shocks=payload.shocks,
        )
        summary = summarize_stress_scenario(
            portfolio_returns,
            confidence_level=payload.confidence_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno calculando stress de portafolio.") from exc

    return PortfolioStressResponse(
        mean_return=float(summary["mean_return"]),
        min_return=float(summary["min_return"]),
        max_drawdown=float(summary["max_drawdown"]),
        var=float(summary["var"]),
        cvar=float(summary["cvar"]),
        observations=len(portfolio_returns),
    )


@router.post("/combined-scenario", response_model=CombinedStressScenarioResponse)
def calculate_combined_stress_scenario(
    payload: CombinedStressScenarioRequest,
) -> CombinedStressScenarioResponse:
    """Apply a combined adverse stress scenario to prices, returns, and bonds."""
    try:
        scenario = combined_adverse_scenario(
            prices=payload.prices,
            returns=payload.returns,
            weights=payload.weights,
            price_shocks=payload.price_shock,
            volatility_multiplier=payload.volatility_multiplier,
            bond_price=payload.bond_price,
            modified_duration=payload.modified_duration,
            convexity=payload.convexity,
            rate_shock=payload.rate_shock,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno calculando escenario combinado.") from exc

    scenario_summary = _to_jsonable(scenario.get("scenario_summary"))
    shocked_prices = _to_jsonable(scenario.get("shocked_prices"))
    shocked_returns = _to_jsonable(scenario.get("shocked_returns"))
    shocked_bond_price = scenario.get("shocked_bond_price")

    return CombinedStressScenarioResponse(
        scenario_summary=scenario_summary,
        shocked_bond_price=float(shocked_bond_price) if shocked_bond_price is not None else None,
        shocked_prices=shocked_prices,
        shocked_returns=shocked_returns,
    )

