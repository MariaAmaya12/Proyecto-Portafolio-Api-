"""FastAPI routes for volatility models."""

from fastapi import APIRouter, HTTPException

from backend.schemas.volatility import EWMAVolatilityRequest, EWMAVolatilityResponse
from src.volatility import ewma_variance, ewma_volatility


router = APIRouter(
    prefix="/volatility",
    tags=["volatility"],
)


@router.post("/ewma", response_model=EWMAVolatilityResponse)
def calculate_ewma_volatility(payload: EWMAVolatilityRequest) -> EWMAVolatilityResponse:
    """Calculate EWMA variance and volatility for a return series."""
    try:
        variance = ewma_variance(payload.returns, lambda_=payload.lambda_)
        volatility = ewma_volatility(
            payload.returns,
            lambda_=payload.lambda_,
            annualize=payload.annualize,
            periods_per_year=payload.periods_per_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno calculando volatilidad EWMA.") from exc

    return EWMAVolatilityResponse(
        ewma_volatility=volatility,
        ewma_variance=variance,
        lambda_=payload.lambda_,
        annualize=payload.annualize,
        periods_per_year=payload.periods_per_year,
        observations=len(payload.returns),
    )

