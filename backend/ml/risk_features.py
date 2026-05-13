from __future__ import annotations

import math

import numpy as np
import pandas as pd

RISK_FEATURE_NAMES = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "rsi",
    "macd_hist",
    "bb_position",
    "close_over_sma20",
    "drawdown_20d",
]

_VALIDATION_RULES: dict[str, tuple] = {
    "rsi":         (lambda v: 0.0 <= v <= 100.0, "'rsi' debe estar entre 0 y 100."),
    "bb_position": (lambda v: -1.0 <= v <= 2.0,  "'bb_position' debe estar entre -1 y 2."),
    "vol_5d":      (lambda v: v >= 0.0,           "'vol_5d' debe ser mayor o igual a 0."),
    "vol_20d":     (lambda v: v >= 0.0,           "'vol_20d' debe ser mayor o igual a 0."),
}


def validate_risk_input(data: dict) -> dict:
    result: dict[str, float] = {}

    for name in RISK_FEATURE_NAMES:
        if name not in data:
            raise ValueError(f"Feature '{name}' faltante en el input.")
        try:
            fval = float(data[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' debe ser un valor numérico.")
        if not math.isfinite(fval):
            raise ValueError(f"'{name}' debe ser un número finito.")
        result[name] = fval

    for name, (rule, msg) in _VALIDATION_RULES.items():
        if not rule(result[name]):
            raise ValueError(msg)

    return result


def risk_features_to_array(features: dict) -> list[list[float]]:
    return [[features[name] for name in RISK_FEATURE_NAMES]]


def make_risk_training_dataset(
    n_steps: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    # Two-regime simulation: 85% normal, 15% stress
    regime = rng.choice([0, 1], size=n_steps, p=[0.85, 0.15])
    mu = np.where(regime == 0, 0.0003, -0.002)
    sigma = np.where(regime == 0, 0.010, 0.025)
    daily_returns = rng.normal(mu, sigma)

    prices = pd.Series(100.0 * np.cumprod(1 + daily_returns))
    r = pd.Series(daily_returns)

    # ── Features: all computed using data ≤ t ──────────────────────────────

    ret_1d = r
    ret_5d = prices.pct_change(5)
    ret_20d = prices.pct_change(20)

    vol_5d = r.rolling(5).std(ddof=1)
    vol_20d = r.rolling(20).std(ddof=1)

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    # Normalize by price to make scale-invariant across different price levels
    macd_hist_series = (macd_line - signal_line) / prices.replace(0, np.nan)

    bb_mid = prices.rolling(20).mean()
    bb_std_s = prices.rolling(20).std(ddof=1)
    bb_upper = bb_mid + 2.0 * bb_std_s
    bb_lower = bb_mid - 2.0 * bb_std_s
    bb_range = (bb_upper - bb_lower).clip(lower=1e-8)
    bb_position_series = (prices - bb_lower) / bb_range

    sma20 = prices.rolling(20).mean()
    close_over_sma20_series = prices / sma20.replace(0, np.nan) - 1.0

    rolling_max_20 = prices.rolling(20).max()
    drawdown_20d_series = prices / rolling_max_20.replace(0, np.nan) - 1.0

    feature_df = pd.DataFrame({
        "ret_1d":           ret_1d,
        "ret_5d":           ret_5d,
        "ret_20d":          ret_20d,
        "vol_5d":           vol_5d,
        "vol_20d":          vol_20d,
        "rsi":              rsi_series,
        "macd_hist":        macd_hist_series,
        "bb_position":      bb_position_series,
        "close_over_sma20": close_over_sma20_series,
        "drawdown_20d":     drawdown_20d_series,
    })

    # ── Target: future 5-day cumulative return (NO leakage: it is the label) ──
    # forward_ret_5d[t] = prices[t+5] / prices[t] - 1  (future, used only as label)
    forward_ret_5d = prices.pct_change(5).shift(-5)
    labels = (forward_ret_5d < -0.02).astype(int)

    full_df = feature_df.copy()
    full_df["_label"] = labels
    full_df = full_df.dropna()

    X = full_df[RISK_FEATURE_NAMES].to_numpy(dtype=float)
    y = full_df["_label"].to_numpy(dtype=int)

    return X, y
