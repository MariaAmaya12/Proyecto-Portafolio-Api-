from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from backend.ml.risk_features import (
    RISK_FEATURE_NAMES,
    risk_features_to_array,
    validate_risk_input,
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "risk_classifier.joblib"


class RiskScorePredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo de riesgo no encontrado en '{MODEL_PATH}'. "
                "Ejecuta 'python -m backend.ml.risk_train' para generarlo."
            )

        artifact = joblib.load(MODEL_PATH)

        for key in ("model", "scaler", "feature_names", "model_version", "horizon_days"):
            if key not in artifact:
                raise ValueError(
                    f"El artifact '{MODEL_PATH}' no contiene la clave requerida '{key}'."
                )

        if artifact["feature_names"] != RISK_FEATURE_NAMES:
            raise ValueError(
                f"feature_names del modelo no coinciden con RISK_FEATURE_NAMES. "
                f"Esperado: {RISK_FEATURE_NAMES}. "
                f"Encontrado: {artifact['feature_names']}."
            )

        if not hasattr(artifact["model"], "predict_proba"):
            raise ValueError(
                "El modelo cargado no soporta predict_proba. "
                "Se requiere un clasificador con estimación de probabilidad."
            )

        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_names: list[str] = artifact["feature_names"]
        self.model_version: str = artifact["model_version"]
        self.horizon_days: int = artifact["horizon_days"]

    def predict(self, payload: dict) -> dict:
        features = validate_risk_input(payload)
        X = risk_features_to_array(features)
        X_sc = self.scaler.transform(X)

        proba = self.model.predict_proba(X_sc)[0]
        classes = list(self.model.classes_)
        risk_score = float(proba[classes.index(1)])

        if risk_score > 0.60:
            risk_level = "alto"
        elif risk_score >= 0.35:
            risk_level = "moderado"
        else:
            risk_level = "bajo"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "horizon_days": self.horizon_days,
            "model_version": self.model_version,
        }


@lru_cache(maxsize=1)
def get_risk_predictor() -> RiskScorePredictor:
    return RiskScorePredictor()
