"""FastAPI routes for option pricing models."""

from fastapi import APIRouter, HTTPException

from backend.schemas.options import BlackScholesRequest, BlackScholesResponse, GreeksResponse
from src.options import (
    black_scholes_call,
    black_scholes_put,
    delta_call,
    delta_put,
    gamma,
    rho_call,
    rho_put,
    theta_call,
    theta_put,
    vega,
)


router = APIRouter(
    prefix="/options",
    tags=["options"],
)


@router.post("/black-scholes", response_model=BlackScholesResponse)
def calculate_black_scholes(payload: BlackScholesRequest) -> BlackScholesResponse:
    """Calculate Black-Scholes call and put prices."""
    try:
        call_price = black_scholes_call(
            spot=payload.spot,
            strike=payload.strike,
            rate=payload.rate,
            volatility=payload.volatility,
            time_to_maturity=payload.time_to_maturity,
        )
        put_price = black_scholes_put(
            spot=payload.spot,
            strike=payload.strike,
            rate=payload.rate,
            volatility=payload.volatility,
            time_to_maturity=payload.time_to_maturity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno calculando precios Black-Scholes.") from exc

    return BlackScholesResponse(call_price=call_price, put_price=put_price)


@router.post("/greeks", response_model=GreeksResponse)
def calculate_greeks(payload: BlackScholesRequest) -> GreeksResponse:
    """Calculate Black-Scholes Greeks for call and put options."""
    try:
        input_values = {
            "spot": payload.spot,
            "strike": payload.strike,
            "rate": payload.rate,
            "volatility": payload.volatility,
            "time_to_maturity": payload.time_to_maturity,
        }
        values = {
            "delta_call": delta_call(**input_values),
            "delta_put": delta_put(**input_values),
            "gamma": gamma(**input_values),
            "vega": vega(**input_values),
            "theta_call": theta_call(**input_values),
            "theta_put": theta_put(**input_values),
            "rho_call": rho_call(**input_values),
            "rho_put": rho_put(**input_values),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno calculando griegas Black-Scholes.") from exc

    return GreeksResponse(**values)

