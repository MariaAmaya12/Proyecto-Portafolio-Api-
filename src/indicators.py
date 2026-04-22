from __future__ import annotations

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame, sma_window: int = 20, ema_window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out[f"SMA_{sma_window}"] = out["Close"].rolling(sma_window).mean()
    out[f"EMA_{ema_window}"] = out["Close"].ewm(span=ema_window, adjust=False).mean()
    return out


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return out


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    return out


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    mean = out["Close"].rolling(window).mean()
    std = out["Close"].rolling(window).std(ddof=1)
    out["BB_mid"] = mean
    out["BB_up"] = mean + n_std * std
    out["BB_low"] = mean - n_std * std
    return out


def add_stochastic(df: pd.DataFrame, window: int = 14, smooth_d: int = 3) -> pd.DataFrame:
    out = df.copy()
    low_n = out["Low"].rolling(window).min()
    high_n = out["High"].rolling(window).max()
    denom = (high_n - low_n).replace(0, np.nan)
    out["%K"] = 100 * (out["Close"] - low_n) / denom
    out["%D"] = out["%K"].rolling(smooth_d).mean()
    return out


def compute_all_indicators(
    df: pd.DataFrame,
    sma_window: int = 20,
    ema_window: int = 20,
    rsi_window: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    stoch_window: int = 14,
) -> pd.DataFrame:
    out = df.copy()
    out = add_moving_averages(out, sma_window=sma_window, ema_window=ema_window)
    out = add_rsi(out, window=rsi_window)
    out = add_macd(out)
    out = add_bollinger_bands(out, window=bb_window, n_std=bb_std)
    out = add_stochastic(out, window=stoch_window)

    price_col = "Adj Close" if "Adj Close" in out.columns else "Close"
    out["SMA_50"] = out[price_col].rolling(50).mean()
    out["SMA_200"] = out[price_col].rolling(200).mean()

    out["RSI"] = out[f"RSI_{rsi_window}"]
    out["BB_upper"] = out["BB_up"]
    out["BB_lower"] = out["BB_low"]
    out["STOCH_K"] = out["%K"]
    out["STOCH_D"] = out["%D"]
    return out
