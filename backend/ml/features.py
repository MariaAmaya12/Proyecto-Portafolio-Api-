from __future__ import annotations

import math

import numpy as np

FEATURE_NAMES = ["close", "sma", "ema", "rsi"]


def validate_feature_values(
    close: float,
    sma: float,
    ema: float,
    rsi: float,
) -> dict:
    raw = {"close": close, "sma": sma, "ema": ema, "rsi": rsi}
    result: dict[str, float] = {}

    for name, val in raw.items():
        try:
            fval = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' debe ser un valor numérico.")
        if not math.isfinite(fval):
            raise ValueError(f"'{name}' debe ser un número finito.")
        result[name] = fval

    for name in ("close", "sma", "ema"):
        if result[name] <= 0.0:
            raise ValueError(f"'{name}' debe ser mayor que 0.")

    if not (0.0 <= result["rsi"] <= 100.0):
        raise ValueError("'rsi' debe estar entre 0 y 100.")

    return result


def features_to_array(features: dict) -> list[list[float]]:
    return [[features[name] for name in FEATURE_NAMES]]


def make_training_dataset(
    n_samples: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    close = rng.uniform(50.0, 200.0, n_samples)
    sma = rng.uniform(50.0, 200.0, n_samples)
    ema = rng.uniform(50.0, 200.0, n_samples)
    rsi = rng.uniform(0.0, 100.0, n_samples)

    X = np.column_stack([close, sma, ema, rsi])

    labels: list[str] = []
    for i in range(n_samples):
        if close[i] > sma[i] and close[i] > ema[i] and 45.0 <= rsi[i] <= 70.0:
            labels.append("Alcista")
        elif (close[i] < sma[i] and close[i] < ema[i]) or rsi[i] > 75.0:
            labels.append("Bajista")
        else:
            labels.append("Neutral")

    return X, np.array(labels)
