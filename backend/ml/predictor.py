from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from backend.ml.features import features_to_array, validate_feature_values

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "signal_classifier.joblib"


class SignalPredictor:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo ML no encontrado en '{MODEL_PATH}'. "
                "Ejecuta 'python -m backend.ml.train' para generarlo."
            )

        artifact: dict = joblib.load(MODEL_PATH)
        self._model = artifact["model"]
        self.model_version: str = artifact["model_version"]
        self.feature_names: list[str] = artifact["feature_names"]
        self.accuracy: float = artifact["accuracy"]

    def predict(
        self,
        close: float,
        sma: float,
        ema: float,
        rsi: float,
    ) -> dict:
        features = validate_feature_values(close=close, sma=sma, ema=ema, rsi=rsi)
        X = features_to_array(features)

        prediction: str = str(self._model.predict(X)[0])

        probability: float = 0.0
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            classes = list(self._model.classes_)
            idx = classes.index(prediction)
            probability = float(proba[idx])

        return {
            "prediction": prediction,
            "probability": probability,
            "model_version": self.model_version,
        }


@lru_cache(maxsize=1)
def get_predictor() -> SignalPredictor:
    return SignalPredictor()
