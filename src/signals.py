from __future__ import annotations

import pandas as pd


COLUMN_ALIASES = {
    "price": ["Close", "Adj Close"],
    "rsi": ["RSI", "RSI_14"],
    "macd": ["MACD"],
    "macd_signal": ["MACD_signal", "macd_signal", "signal"],
    "bb_upper": ["BB_upper", "BB_up"],
    "bb_lower": ["BB_lower", "BB_low"],
    "sma_50": ["SMA_50"],
    "sma_200": ["SMA_200"],
    "stoch_k": ["STOCH_K", "%K"],
    "stoch_d": ["STOCH_D", "%D"],
}


SIGNAL_REQUIREMENTS = {
    "macd_buy": ["macd", "macd_signal"],
    "macd_sell": ["macd", "macd_signal"],
    "rsi_buy": ["rsi"],
    "rsi_sell": ["rsi"],
    "boll_buy": ["price", "bb_lower"],
    "boll_sell": ["price", "bb_upper"],
    "golden_cross": ["sma_50", "sma_200"],
    "death_cross": ["sma_50", "sma_200"],
    "stoch_buy": ["stoch_k", "stoch_d"],
    "stoch_sell": ["stoch_k", "stoch_d"],
}


def _first_existing(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias

    lower_map = {str(col).lower(): col for col in df.columns}
    for alias in aliases:
        match = lower_map.get(alias.lower())
        if match is not None:
            return match

    return None


def _resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {key: _first_existing(df, aliases) for key, aliases in COLUMN_ALIASES.items()}


def _diagnostic_row(
    signal_name: str,
    active: bool,
    evaluated: bool,
    reason: str,
    columns_used: list[str],
    missing_columns: list[str] | None = None,
) -> dict:
    return {
        "signal": signal_name,
        "evaluated": bool(evaluated),
        "active": bool(active),
        "reason": reason,
        "columns_used": columns_used,
        "missing_columns": missing_columns or [],
    }


def _missing_for_signal(resolved: dict[str, str | None], signal_name: str) -> list[str]:
    return [key for key in SIGNAL_REQUIREMENTS[signal_name] if resolved.get(key) is None]


def _has_current_values(df: pd.DataFrame, columns: list[str]) -> bool:
    return not df.empty and not df.iloc[-1][columns].isna().any()


def _has_last_two_values(df: pd.DataFrame, columns: list[str]) -> bool:
    return len(df) >= 2 and not df.iloc[-2:][columns].isna().any().any()


def evaluate_signal_diagnostics(
    df: pd.DataFrame,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    stoch_overbought: float = 80,
    stoch_oversold: float = 20,
) -> dict:
    """
    Evalua reglas tecnicas sobre el ultimo dato disponible y explica cada resultado.
    """
    if df.empty:
        details = {signal_name: False for signal_name in SIGNAL_REQUIREMENTS}
        diagnostics = [
            _diagnostic_row(
                signal_name,
                active=False,
                evaluated=False,
                reason="No evaluable por datos insuficientes",
                columns_used=[],
            )
            for signal_name in SIGNAL_REQUIREMENTS
        ]
        return {"details": details, "diagnostics": diagnostics, "resolved_columns": {}}

    resolved = _resolve_columns(df)
    details = {}
    diagnostics = []

    def evaluate_current(signal_name: str, condition, keys: list[str]):
        missing = _missing_for_signal(resolved, signal_name)
        columns = [resolved[key] for key in keys if resolved.get(key) is not None]

        if missing:
            details[signal_name] = False
            diagnostics.append(
                _diagnostic_row(signal_name, False, False, "columna faltante", columns, missing)
            )
            return

        if not _has_current_values(df, columns):
            details[signal_name] = False
            diagnostics.append(
                _diagnostic_row(
                    signal_name,
                    False,
                    False,
                    "No evaluable por datos insuficientes",
                    columns,
                )
            )
            return

        last = df.iloc[-1]
        active = bool(condition(last))
        details[signal_name] = active
        diagnostics.append(
            _diagnostic_row(
                signal_name,
                active,
                True,
                "señal activa" if active else "condición no cumplida",
                columns,
            )
        )

    def evaluate_cross(signal_name: str, condition, keys: list[str]):
        missing = _missing_for_signal(resolved, signal_name)
        columns = [resolved[key] for key in keys if resolved.get(key) is not None]

        if missing:
            details[signal_name] = False
            diagnostics.append(
                _diagnostic_row(signal_name, False, False, "columna faltante", columns, missing)
            )
            return

        if not _has_last_two_values(df, columns):
            details[signal_name] = False
            diagnostics.append(
                _diagnostic_row(
                    signal_name,
                    False,
                    False,
                    "No evaluable por datos insuficientes",
                    columns,
                )
            )
            return

        prev = df.iloc[-2]
        last = df.iloc[-1]
        active = bool(condition(prev, last))
        details[signal_name] = active
        diagnostics.append(
            _diagnostic_row(
                signal_name,
                active,
                True,
                "señal activa" if active else "condición no cumplida",
                columns,
            )
        )

    macd = resolved["macd"]
    macd_signal = resolved["macd_signal"]
    rsi = resolved["rsi"]
    price = resolved["price"]
    bb_upper = resolved["bb_upper"]
    bb_lower = resolved["bb_lower"]
    sma_50 = resolved["sma_50"]
    sma_200 = resolved["sma_200"]
    stoch_k = resolved["stoch_k"]
    stoch_d = resolved["stoch_d"]

    evaluate_cross(
        "macd_buy",
        lambda prev, last: (prev[macd] <= prev[macd_signal]) and (last[macd] > last[macd_signal]),
        ["macd", "macd_signal"],
    )
    evaluate_cross(
        "macd_sell",
        lambda prev, last: (prev[macd] >= prev[macd_signal]) and (last[macd] < last[macd_signal]),
        ["macd", "macd_signal"],
    )
    evaluate_current("rsi_buy", lambda last: last[rsi] < rsi_oversold, ["rsi"])
    evaluate_current("rsi_sell", lambda last: last[rsi] > rsi_overbought, ["rsi"])
    evaluate_current("boll_buy", lambda last: last[price] <= last[bb_lower], ["price", "bb_lower"])
    evaluate_current("boll_sell", lambda last: last[price] >= last[bb_upper], ["price", "bb_upper"])
    evaluate_cross(
        "golden_cross",
        lambda prev, last: (prev[sma_50] <= prev[sma_200]) and (last[sma_50] > last[sma_200]),
        ["sma_50", "sma_200"],
    )
    evaluate_cross(
        "death_cross",
        lambda prev, last: (prev[sma_50] >= prev[sma_200]) and (last[sma_50] < last[sma_200]),
        ["sma_50", "sma_200"],
    )
    evaluate_cross(
        "stoch_buy",
        lambda prev, last: (
            (prev[stoch_k] <= prev[stoch_d])
            and (last[stoch_k] > last[stoch_d])
            and (last[stoch_k] <= stoch_oversold)
            and (last[stoch_d] <= stoch_oversold)
        ),
        ["stoch_k", "stoch_d"],
    )
    evaluate_cross(
        "stoch_sell",
        lambda prev, last: (
            (prev[stoch_k] >= prev[stoch_d])
            and (last[stoch_k] < last[stoch_d])
            and (last[stoch_k] >= stoch_overbought)
            and (last[stoch_d] >= stoch_overbought)
        ),
        ["stoch_k", "stoch_d"],
    )

    return {"details": details, "diagnostics": diagnostics, "resolved_columns": resolved}


def evaluate_signals(
    df: pd.DataFrame,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    stoch_overbought: float = 80,
    stoch_oversold: float = 20,
) -> dict:
    """
    Evalua reglas tecnicas usando el ultimo dato disponible.
    """
    evaluation = evaluate_signal_diagnostics(
        df,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        stoch_overbought=stoch_overbought,
        stoch_oversold=stoch_oversold,
    )
    flags = evaluation["details"]

    score_buy = (
        int(flags["macd_buy"])
        + int(flags["rsi_buy"])
        + int(flags["boll_buy"])
        + int(flags["golden_cross"])
        + int(flags["stoch_buy"])
    )
    score_sell = (
        int(flags["macd_sell"])
        + int(flags["rsi_sell"])
        + int(flags["boll_sell"])
        + int(flags["death_cross"])
        + int(flags["stoch_sell"])
    )

    if score_buy >= 3:
        recommendation = "Compra"
        color = "green"
    elif score_sell >= 3:
        recommendation = "Venta"
        color = "red"
    else:
        recommendation = "Mantener"
        color = "orange"

    reasons = []
    if flags["macd_buy"]:
        reasons.append("MACD alcista")
    if flags["macd_sell"]:
        reasons.append("MACD bajista")
    if flags["rsi_buy"]:
        reasons.append("RSI en sobreventa")
    if flags["rsi_sell"]:
        reasons.append("RSI en sobrecompra")
    if flags["boll_buy"]:
        reasons.append("Precio toca Bollinger inferior")
    if flags["boll_sell"]:
        reasons.append("Precio toca Bollinger superior")
    if flags["golden_cross"]:
        reasons.append("Golden cross")
    if flags["death_cross"]:
        reasons.append("Death cross")
    if flags["stoch_buy"]:
        reasons.append("Estocastico alcista en zona extrema")
    if flags["stoch_sell"]:
        reasons.append("Estocastico bajista en zona extrema")

    return {
        "score_buy": score_buy,
        "score_sell": score_sell,
        "recommendation": recommendation,
        "color": color,
        "reasons": reasons,
        "details": flags,
        "diagnostics": evaluation["diagnostics"],
        "resolved_columns": evaluation["resolved_columns"],
    }
