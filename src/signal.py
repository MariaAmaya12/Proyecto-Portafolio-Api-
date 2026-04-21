from __future__ import annotations


def compute_signal(close: float, sma: float, ema: float, rsi: float) -> str:
    if close > sma and close > ema and rsi >= 55:
        return "Alcista"
    if close < sma and close < ema and rsi <= 45:
        return "Bajista"
    return "Neutral"
