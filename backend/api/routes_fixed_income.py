"""FastAPI routes for fixed income models."""

from fastapi import APIRouter, HTTPException

from backend.schemas.fixed_income import (
    BondMetricsRequest,
    BondMetricsResponse,
    NelsonSiegelRequest,
    NelsonSiegelResponse,
)
from src.fixed_income import (
    bond_price,
    convexity,
    macaulay_duration,
    modified_duration,
    nelson_siegel_yield,
)


router = APIRouter(
    prefix="/fixed-income",
    tags=["fixed-income"],
)


@router.post("/bond-metrics", response_model=BondMetricsResponse)
def calculate_bond_metrics(payload: BondMetricsRequest) -> BondMetricsResponse:
    """Calculate price, duration, and convexity for a coupon bond."""
    try:
        price = bond_price(
            face_value=payload.face_value,
            coupon_rate=payload.coupon_rate,
            market_rate=payload.market_rate,
            maturity_years=payload.maturity_years,
            frequency=payload.frequency,
        )
        macaulay = macaulay_duration(
            face_value=payload.face_value,
            coupon_rate=payload.coupon_rate,
            market_rate=payload.market_rate,
            maturity_years=payload.maturity_years,
            frequency=payload.frequency,
        )
        modified = modified_duration(
            face_value=payload.face_value,
            coupon_rate=payload.coupon_rate,
            market_rate=payload.market_rate,
            maturity_years=payload.maturity_years,
            frequency=payload.frequency,
        )
        bond_convexity = convexity(
            face_value=payload.face_value,
            coupon_rate=payload.coupon_rate,
            market_rate=payload.market_rate,
            maturity_years=payload.maturity_years,
            frequency=payload.frequency,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno calculando metricas de bono.") from exc

    return BondMetricsResponse(
        price=price,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=bond_convexity,
    )


@router.post("/nelson-siegel", response_model=NelsonSiegelResponse)
def calculate_nelson_siegel(payload: NelsonSiegelRequest) -> NelsonSiegelResponse:
    """Calculate Nelson-Siegel yields for the requested maturities."""
    try:
        yields = nelson_siegel_yield(
            payload.maturities,
            beta0=payload.beta0,
            beta1=payload.beta1,
            beta2=payload.beta2,
            tau=payload.tau,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Error interno calculando curva Nelson-Siegel.") from exc

    return NelsonSiegelResponse(
        maturities=payload.maturities,
        yields=[float(value) for value in yields],
    )

