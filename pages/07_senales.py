import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.date_utils import yfinance_exclusive_end
from src.download import data_error_message, download_single_ticker
from src.indicators import compute_all_indicators
from src.signals import evaluate_signals
from src.ui_components import render_explanation_expander, render_section, render_table
from src.ui_navigation import render_sidebar_navigation
from src.ui_style import apply_global_typography, render_page_title

ensure_project_dirs()
apply_global_typography()
render_sidebar_navigation()
render_page_title(
    "Módulo 7 - Señales y alertas",
    "Resume señales técnicas por activo y las traduce en una lectura operativa más clara.",
)


SIGNAL_LABELS = {
    "macd_buy": "MACD compra",
    "macd_sell": "MACD venta",
    "rsi_buy": "RSI sobreventa",
    "rsi_sell": "RSI sobrecompra",
    "boll_buy": "Bollinger compra",
    "boll_sell": "Bollinger venta",
    "golden_cross": "Golden cross",
    "death_cross": "Death cross",
    "stoch_buy": "Estocástico compra",
    "stoch_sell": "Estocástico venta",
}


# ==============================
# UI helpers
# ==============================
def inject_ui_css():
    st.markdown(
        """
        <style>
        .section-intro-box {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }
        .section-intro-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-intro-subtitle {
            font-size: 0.86rem;
            color: #64748b;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sanitize_text(text):
    if text is None:
        return ""
    return str(text).replace("<", "").replace(">", "")


def signal_card(asset_name, ticker, recommendation, semaforo_estado, semaforo_msg, score_buy, score_sell, active_text, level):
    styles = {
        "positive": {
            "bg": "linear-gradient(180deg, #ecfdf5 0%, #f0fdf4 100%)",
            "border": "rgba(22, 163, 74, 0.22)",
            "pill_bg": "rgba(22, 163, 74, 0.14)",
            "pill_color": "#15803d",
        },
        "warning": {
            "bg": "linear-gradient(180deg, #fffbeb 0%, #fefce8 100%)",
            "border": "rgba(234, 179, 8, 0.28)",
            "pill_bg": "rgba(234, 179, 8, 0.16)",
            "pill_color": "#a16207",
        },
        "danger": {
            "bg": "linear-gradient(180deg, #fff1f2 0%, #fef2f2 100%)",
            "border": "rgba(220, 38, 38, 0.22)",
            "pill_bg": "rgba(220, 38, 38, 0.12)",
            "pill_color": "#b91c1c",
        },
    }

    s = styles.get(level, styles["warning"])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .card {{
                background: {s["bg"]};
                border: 1px solid {s["border"]};
                border-radius: 20px;
                padding: 18px;
                min-height: 250px;
                box-sizing: border-box;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            }}
            .asset {{
                font-size: 1rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.4rem;
            }}
            .ticker {{
                font-size: 0.82rem;
                color: #64748b;
                margin-bottom: 0.8rem;
            }}
            .pill {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                background: {s["pill_bg"]};
                color: {s["pill_color"]};
                font-size: 0.82rem;
                font-weight: 800;
                margin-bottom: 0.8rem;
            }}
            .subtitle {{
                font-size: 0.82rem;
                font-weight: 700;
                color: #334155;
                margin-top: 0.5rem;
                margin-bottom: 0.2rem;
            }}
            .text {{
                font-size: 0.82rem;
                color: #334155;
                line-height: 1.45;
                margin-bottom: 0.4rem;
            }}
            .scores {{
                display: flex;
                gap: 10px;
                margin-top: 0.6rem;
                margin-bottom: 0.4rem;
            }}
            .score-box {{
                flex: 1;
                background: rgba(255,255,255,0.72);
                border: 1px solid rgba(15, 23, 42, 0.06);
                border-radius: 12px;
                padding: 10px 12px;
            }}
            .score-label {{
                font-size: 0.74rem;
                color: #64748b;
                margin-bottom: 0.15rem;
            }}
            .score-value {{
                font-size: 1.05rem;
                font-weight: 800;
                color: #0f172a;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="asset">{sanitize_text(asset_name)}</div>
            <div class="ticker">{sanitize_text(ticker)}</div>
            <div class="pill">{sanitize_text(recommendation)}</div>

            <div class="subtitle">Semáforo</div>
            <div class="text"><strong>{sanitize_text(semaforo_estado)}</strong> · {sanitize_text(semaforo_msg)}</div>

            <div class="scores">
                <div class="score-box">
                    <div class="score-label">Score compra</div>
                    <div class="score-value">{score_buy}</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Score venta</div>
                    <div class="score-value">{score_sell}</div>
                </div>
            </div>

            <div class="subtitle">Señales activas</div>
            <div class="text">{sanitize_text(active_text)}</div>
        </div>
    </body>
    </html>
    """
    components.html(html, height=290)


inject_ui_css()


# ==============================
# Helpers lógicos
# ==============================
def normalize_flags(flags: dict) -> dict:
    return {k: bool(v) for k, v in flags.items()}


def active_signals(flags: dict) -> list[str]:
    clean = normalize_flags(flags)
    return [SIGNAL_LABELS.get(k, k) for k, v in clean.items() if v]


def signal_table(flags: dict) -> pd.DataFrame:
    clean = normalize_flags(flags)
    return pd.DataFrame(
        {
            "Señal": [SIGNAL_LABELS.get(k, k) for k in clean.keys()],
            "Activa": ["Sí" if v else "No" for v in clean.values()],
        }
    )


DIAGNOSTIC_COLUMN_ALIASES = {
    "Close / Adj Close": ["Close", "Adj Close", "close", "adj_close", "adj close"],
    "RSI": ["rsi", "RSI", "RSI_14", "RSI(14)", "rsi_14"],
    "MACD": ["macd", "MACD"],
    "MACD signal": ["macd_signal", "signal", "MACD_signal", "MACD Signal"],
    "Bollinger superior": ["BB_upper", "BB_up", "boll_upper", "bb_upper", "upper_band", "Bollinger Upper"],
    "Bollinger inferior": ["BB_lower", "BB_low", "boll_lower", "bb_lower", "lower_band", "Bollinger Lower"],
    "Media 50": ["sma_50", "SMA_50", "ma_50", "MA_50"],
    "Media 200": ["sma_200", "SMA_200", "ma_200", "MA_200"],
    "Estocástico K": ["STOCH_K", "stoch_k", "Stoch_K", "%K", "stochastic_k"],
    "Estocástico D": ["STOCH_D", "stoch_d", "Stoch_D", "%D", "stochastic_d"],
}


def find_matching_columns(columns, aliases: list[str]) -> list[str]:
    columns_list = list(columns)
    lower_map = {str(col).lower(): col for col in columns_list}
    matches = []

    for alias in aliases:
        direct = lower_map.get(alias.lower())
        if direct is not None and direct not in matches:
            matches.append(direct)

    if matches:
        return matches

    alias_tokens = [
        alias.lower().replace("_", "").replace(" ", "").replace("(", "").replace(")", "")
        for alias in aliases
    ]
    for col in columns_list:
        normalized = str(col).lower().replace("_", "").replace(" ", "").replace("(", "").replace(")", "")
        if any(token and token in normalized for token in alias_tokens):
            matches.append(col)

    return matches


def diagnostic_column_summary(ind: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    rows = []
    existing_columns = []
    last_row = ind.tail(1)

    for label, aliases in DIAGNOSTIC_COLUMN_ALIASES.items():
        matches = find_matching_columns(ind.columns, aliases)
        if not matches:
            rows.append(
                {
                    "Indicador esperado": label,
                    "Columnas encontradas": "NO EXISTE",
                    "NaN en última fila": "NO EXISTE",
                }
            )
            continue

        for col in matches:
            if col not in existing_columns:
                existing_columns.append(col)

        nan_status = {
            str(col): bool(last_row[col].isna().iloc[0])
            for col in matches
            if col in last_row.columns and not last_row.empty
        }
        rows.append(
            {
                "Indicador esperado": label,
                "Columnas encontradas": ", ".join(str(col) for col in matches),
                "NaN en última fila": nan_status if nan_status else "Sin última fila",
            }
        )

    return pd.DataFrame(rows), existing_columns


def build_historical_signal_points(ind: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    if ind is None or ind.empty:
        return pd.DataFrame(columns=["date", "signal", "side", "price"])

    working = ind.copy()

    def first_match(*aliases_groups: list[str]) -> str | None:
        for aliases in aliases_groups:
            matches = find_matching_columns(working.columns, aliases)
            if matches:
                return matches[0]
        return None

    price_col = first_match(
        ["Close", "Adj Close", "close", "adj_close", "adj close"],
    )
    if price_col is None:
        return pd.DataFrame(columns=["date", "signal", "side", "price"])

    price = pd.to_numeric(working[price_col], errors="coerce")
    signal_rows = []

    def append_mask(mask: pd.Series, signal_name: str, side: str):
        if mask is None:
            return
        valid_mask = mask.fillna(False)
        if not valid_mask.any():
            return
        selected = working.loc[valid_mask].copy()
        if selected.empty:
            return
        selected_price = pd.to_numeric(selected[price_col], errors="coerce")
        selected = selected.assign(
            date=selected.index,
            signal=signal_name,
            side=side,
            price=selected_price,
        )[["date", "signal", "side", "price"]]
        selected = selected.dropna(subset=["price"])
        if not selected.empty:
            signal_rows.append(selected)

    macd_col = first_match(["macd", "MACD"])
    macd_signal_col = first_match(["macd_signal", "signal", "MACD_signal", "MACD Signal"])
    if macd_col and macd_signal_col:
        macd = pd.to_numeric(working[macd_col], errors="coerce")
        macd_signal = pd.to_numeric(working[macd_signal_col], errors="coerce")
        append_mask((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)), "MACD compra", "buy")
        append_mask((macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)), "MACD venta", "sell")

    rsi_col = first_match(["rsi", "RSI", "RSI_14", "RSI(14)", "rsi_14"])
    if rsi_col:
        rsi = pd.to_numeric(working[rsi_col], errors="coerce")
        append_mask(
            (rsi <= thresholds["rsi_oversold"]) & (rsi.shift(1) > thresholds["rsi_oversold"]),
            "RSI sobreventa",
            "buy",
        )
        append_mask(
            (rsi >= thresholds["rsi_overbought"]) & (rsi.shift(1) < thresholds["rsi_overbought"]),
            "RSI sobrecompra",
            "sell",
        )

    boll_upper_col = first_match(["BB_upper", "BB_up", "boll_upper", "bb_upper", "upper_band", "Bollinger Upper"])
    boll_lower_col = first_match(["BB_lower", "BB_low", "boll_lower", "bb_lower", "lower_band", "Bollinger Lower"])
    if boll_lower_col:
        boll_lower = pd.to_numeric(working[boll_lower_col], errors="coerce")
        append_mask(
            (price <= boll_lower) & (price.shift(1) > boll_lower.shift(1)),
            "Bollinger compra",
            "buy",
        )
    if boll_upper_col:
        boll_upper = pd.to_numeric(working[boll_upper_col], errors="coerce")
        append_mask(
            (price >= boll_upper) & (price.shift(1) < boll_upper.shift(1)),
            "Bollinger venta",
            "sell",
        )

    sma_50_col = first_match(["sma_50", "SMA_50", "ma_50", "MA_50"])
    sma_200_col = first_match(["sma_200", "SMA_200", "ma_200", "MA_200"])
    if sma_50_col and sma_200_col:
        sma_50 = pd.to_numeric(working[sma_50_col], errors="coerce")
        sma_200 = pd.to_numeric(working[sma_200_col], errors="coerce")
        append_mask((sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1)), "Golden cross", "buy")
        append_mask((sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1)), "Death cross", "sell")

    stoch_k_col = first_match(["STOCH_K", "stoch_k", "Stoch_K", "%K", "stochastic_k"])
    stoch_d_col = first_match(["STOCH_D", "stoch_d", "Stoch_D", "%D", "stochastic_d"])
    if stoch_k_col and stoch_d_col:
        stoch_k = pd.to_numeric(working[stoch_k_col], errors="coerce")
        stoch_d = pd.to_numeric(working[stoch_d_col], errors="coerce")
        append_mask(
            (stoch_k > stoch_d)
            & (stoch_k.shift(1) <= stoch_d.shift(1))
            & (stoch_k <= thresholds["stoch_oversold"]),
            "Estocástico compra",
            "buy",
        )
        append_mask(
            (stoch_k < stoch_d)
            & (stoch_k.shift(1) >= stoch_d.shift(1))
            & (stoch_k >= thresholds["stoch_overbought"]),
            "Estocástico venta",
            "sell",
        )

    if not signal_rows:
        return pd.DataFrame(columns=["date", "signal", "side", "price"])

    signal_points = pd.concat(signal_rows, ignore_index=True)
    signal_points["date"] = pd.to_datetime(signal_points["date"], errors="coerce")
    signal_points["price"] = pd.to_numeric(signal_points["price"], errors="coerce")
    signal_points = signal_points.dropna(subset=["date", "price"]).sort_values(["date", "signal"]).reset_index(drop=True)
    return signal_points


def plot_asset_signals(ind: pd.DataFrame, asset_name: str, ticker: str, signal_points: pd.DataFrame):
    price_matches = find_matching_columns(ind.columns, ["Close", "Adj Close", "close", "adj_close", "adj close"])
    price_col = price_matches[0] if price_matches else None
    if price_col is None:
        fig = go.Figure()
        fig.update_layout(title=f"Señales históricas - {asset_name}")
        return fig

    price_series = pd.to_numeric(ind[price_col], errors="coerce")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ind.index,
            y=price_series,
            mode="lines",
            name=f"Precio ({ticker})",
            line=dict(color="#1d4ed8", width=2),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>",
        )
    )

    if signal_points is not None and not signal_points.empty:
        buy_points = signal_points.loc[signal_points["side"] == "buy"]
        sell_points = signal_points.loc[signal_points["side"] == "sell"]

        if not buy_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_points["date"],
                    y=buy_points["price"],
                    mode="markers",
                    name="Señales de compra",
                    marker=dict(color="#16a34a", size=10, symbol="triangle-up"),
                    customdata=buy_points["signal"],
                    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Señal: %{customdata}<br>Precio: %{y:.2f}<extra></extra>",
                )
            )

        if not sell_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_points["date"],
                    y=sell_points["price"],
                    mode="markers",
                    name="Señales de venta",
                    marker=dict(color="#dc2626", size=10, symbol="triangle-down"),
                    customdata=sell_points["signal"],
                    hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Señal: %{customdata}<br>Precio: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"Señales históricas - {asset_name}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        hovermode="closest",
        legend_title="Serie",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render_diagnostic(
    asset_name: str,
    ticker: str,
    start_date,
    end_date,
    df: pd.DataFrame,
    ind: pd.DataFrame | None = None,
    signal: dict | None = None,
):
    with st.expander(f"Diagnóstico técnico (por activo) - {asset_name} ({ticker})"):
        st.markdown("**Diagnóstico de descarga**")
        st.write(
            {
                "start_date_usuario": str(start_date),
                "end_date_usuario": str(end_date),
                "end_date_enviado_a_yfinance": yfinance_exclusive_end(str(end_date)),
                "filas_descargadas": len(df),
                "fecha_min": df.index.min() if not df.empty else None,
                "fecha_max": df.index.max() if not df.empty else None,
                "ultima_fecha_descargada": df.index.max() if not df.empty else None,
                "shape": df.shape,
                "columnas_descargadas": list(df.columns),
            }
        )

        if ind is None:
            st.warning("No hay DataFrame de indicadores para este activo porque la descarga llegó vacía.")
            return

        st.markdown("**Columnas reales en indicadores**")
        st.write(list(ind.columns))

        summary, key_columns = diagnostic_column_summary(ind)
        st.markdown("**Estado de columnas clave**")
        render_table(summary, hide_index=True, width="stretch")

        missing_critical = summary.loc[
            summary["Columnas encontradas"].eq("NO EXISTE"),
            "Indicador esperado",
        ].tolist()
        if missing_critical:
            st.warning(f"Columnas críticas faltantes: {', '.join(missing_critical)}")

        if key_columns:
            st.markdown("**Últimas 5 filas de columnas clave existentes**")
            render_table(ind[key_columns].tail(5), hide_index=True, width="stretch")
        else:
            st.warning("No se encontró ninguna columna clave esperada en el DataFrame de indicadores.")

        if signal and signal.get("diagnostics"):
            st.markdown("**Evaluación de señales**")
            signal_diag = pd.DataFrame(signal["diagnostics"])
            signal_diag["Señal"] = signal_diag["signal"].map(SIGNAL_LABELS).fillna(signal_diag["signal"])
            render_table(
                signal_diag[
                    [
                        "Señal",
                        "evaluated",
                        "active",
                        "reason",
                        "columns_used",
                        "missing_columns",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        if len(ind.columns) <= 20:
            st.markdown("**Últimas 5 filas completas**")
            render_table(ind.tail(5), hide_index=True, width="stretch")
        else:
            st.caption(
                f"Se omite ind.tail(5) completo porque el DataFrame tiene {len(ind.columns)} columnas."
            )


def classify_signal_risk(signal: dict) -> dict:
    recommendation = str(signal.get("recommendation", "")).lower()
    score_buy = int(signal.get("score_buy", 0))
    score_sell = int(signal.get("score_sell", 0))
    reasons = signal.get("reasons", []) or []

    if "compra" in recommendation or score_buy >= score_sell + 2:
        return {
            "estado": "Favorable",
            "ui": "positive",
            "mensaje": (
                "Predominan señales de fortaleza o entrada táctica de corto plazo."
            ),
        }

    if "venta" in recommendation or score_sell >= score_buy + 2:
        return {
            "estado": "Desfavorable",
            "ui": "danger",
            "mensaje": (
                "Predominan señales de deterioro técnico, agotamiento o presión bajista."
            ),
        }

    if len(reasons) == 0 and score_buy == score_sell:
        return {
            "estado": "Neutral",
            "ui": "warning",
            "mensaje": (
                "No hay una ventaja técnica clara; el activo luce en zona de espera."
            ),
        }

    return {
        "estado": "Mixto / Precaución",
        "ui": "warning",
        "mensaje": (
            "Existen señales mixtas o poco concluyentes entre tendencia, momentum y reversión."
        ),
    }


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros")

    horizonte = st.selectbox(
        "Horizonte de análisis",
        [
            "1 mes",
            "Trimestre",
            "Semestre",
            "1 año",
            "2 años",
            "3 años",
            "5 años",
            "Personalizado",
        ],
        index=3,
    )

    fecha_fin_ref = pd.to_datetime(DEFAULT_END_DATE)

    if horizonte == "1 mes":
        start_date = (fecha_fin_ref - pd.DateOffset(months=1)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "Trimestre":
        start_date = (fecha_fin_ref - pd.DateOffset(months=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "Semestre":
        start_date = (fecha_fin_ref - pd.DateOffset(months=6)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "1 año":
        start_date = (fecha_fin_ref - pd.DateOffset(years=1)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "2 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=2)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "3 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "5 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=5)).date()
        end_date = fecha_fin_ref.date()
    else:
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="sig_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="sig_end")

    st.divider()
    with st.expander("Umbrales configurables"):
        rsi_overbought = st.slider("RSI sobrecompra", min_value=60, max_value=90, value=70)
        rsi_oversold = st.slider("RSI sobreventa", min_value=10, max_value=40, value=30)
        stoch_overbought = st.slider("Estocástico sobrecompra", min_value=60, max_value=95, value=80)
        stoch_oversold = st.slider("Estocástico sobreventa", min_value=5, max_value=40, value=20)

    st.divider()
    modo_diagnostico = st.checkbox("Modo diagnóstico (temporal)", value=False)

    if st.button("Actualizar datos", key="sig_refresh_data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ==============================
# Introducción
# ==============================
render_section(
    "Cómo leer este módulo",
    "Cada activo se resume en una tarjeta con recomendación, semáforo técnico, score de compra/venta y señales activas más relevantes.",
)

render_explanation_expander(
    "Cómo interpretar el semáforo técnico",
    [
        "Favorable: predominan señales de entrada o fortaleza técnica.",
        "Neutral / Precaución: señales mixtas o sin ventaja clara.",
        "Desfavorable: predominan señales de venta, agotamiento o deterioro técnico.",
        "Estas señales son tácticas y deben complementarse con riesgo, CAPM, VaR/CVaR y benchmark.",
    ],
)


# ==============================
# Construcción de tarjetas
# ==============================
cards_data = []

for asset_name, meta in ASSETS.items():
    ticker = meta["ticker"]
    df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))

    if df.empty:
        if modo_diagnostico:
            render_diagnostic(asset_name, ticker, start_date, end_date, df)
        continue

    ind = compute_all_indicators(df)
    signal = evaluate_signals(
        ind,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        stoch_overbought=stoch_overbought,
        stoch_oversold=stoch_oversold,
    )
    if modo_diagnostico:
        render_diagnostic(asset_name, ticker, start_date, end_date, df, ind, signal)

    if not signal:
        continue

    flags = normalize_flags(signal["details"])
    active = active_signals(flags)
    semaforo = classify_signal_risk(signal)

    cards_data.append(
        {
            "asset_name": asset_name,
            "ticker": ticker,
            "df": df,
            "ind": ind,
            "recommendation": signal.get("recommendation", "Sin recomendación"),
            "semaforo_estado": semaforo["estado"],
            "semaforo_msg": semaforo["mensaje"],
            "score_buy": signal.get("score_buy", 0),
            "score_sell": signal.get("score_sell", 0),
            "active_text": ", ".join(active) if active else "Ninguna",
            "flags": flags,
            "ui": semaforo["ui"],
        }
    )

if not cards_data:
    st.warning(data_error_message("No fue posible construir señales para los activos en la ventana seleccionada."))
    st.stop()


# ==============================
# Gráfico de señales por activo
# ==============================
st.markdown("### Gráfico de señales por activo")
render_section(
    "Señales sobre el precio",
    "El gráfico ubica señales históricas de compra y venta sobre la serie de precios del activo seleccionado.",
)

chart_asset_options = [item["asset_name"] for item in cards_data]
selected_chart_asset = st.selectbox("Activo para gráfico", chart_asset_options)
selected_chart_item = next(item for item in cards_data if item["asset_name"] == selected_chart_asset)

signal_thresholds = {
    "rsi_overbought": rsi_overbought,
    "rsi_oversold": rsi_oversold,
    "stoch_overbought": stoch_overbought,
    "stoch_oversold": stoch_oversold,
}
signal_points = build_historical_signal_points(selected_chart_item["ind"], signal_thresholds)

st.plotly_chart(
    plot_asset_signals(
        ind=selected_chart_item["ind"],
        asset_name=selected_chart_item["asset_name"],
        ticker=selected_chart_item["ticker"],
        signal_points=signal_points,
    ),
    width="stretch",
)

render_explanation_expander(
    "Cómo interpretar el gráfico de señales",
    [
        "Los marcadores de compra indican activación histórica de señales técnicas alcistas.",
        "Los marcadores de venta indican activación histórica de señales técnicas bajistas.",
        "Varias señales cercanas en el tiempo pueden indicar confirmación técnica.",
        "Señales aisladas deben interpretarse con cautela.",
        "El gráfico es táctico y debe complementarse con riesgo, CAPM, VaR/CVaR y benchmark.",
    ],
)


# ==============================
# Grid 2 columnas
# ==============================
st.markdown("### Lectura por activo")

for i in range(0, len(cards_data), 2):
    cols = st.columns(2)
    row_items = cards_data[i:i + 2]

    for col, item in zip(cols, row_items):
        with col:
            signal_card(
                asset_name=item["asset_name"],
                ticker=item["ticker"],
                recommendation=item["recommendation"],
                semaforo_estado=item["semaforo_estado"],
                semaforo_msg=item["semaforo_msg"],
                score_buy=item["score_buy"],
                score_sell=item["score_sell"],
                active_text=item["active_text"],
                level=item["ui"],
            )

            with st.expander(f"Ver detalle de señales - {item['asset_name']}"):
                render_table(
                    signal_table(item["flags"]),
                    hide_index=True,
                    width="stretch",
                )


# ==============================
# Interpretación breve final
# ==============================
favorables = sum(1 for x in cards_data if x["semaforo_estado"] == "Favorable")
desfavorables = sum(1 for x in cards_data if x["semaforo_estado"] == "Desfavorable")
mixtas = len(cards_data) - favorables - desfavorables

st.markdown("### Interpretación breve")

if favorables > desfavorables:
    st.success(
        f"""
        En esta ventana predominan señales **favorables** ({favorables} activos). La lectura técnica luce más constructiva
        que defensiva, aunque conviene contrastarla con riesgo, benchmark y contexto macro.
        """
    )
elif desfavorables > favorables:
    st.error(
        f"""
        En esta ventana predominan señales **desfavorables** ({desfavorables} activos). La lectura sugiere mayor cautela,
        porque varios activos muestran presión bajista, pérdida de fuerza o zonas de posible agotamiento.
        """
    )
else:
    st.warning(
        f"""
        La lectura agregada está **empatada**: hay {favorables} activos favorables, {desfavorables} desfavorables
        y {mixtas} neutrales o mixtos. En este caso conviene revisar activo por activo antes de tomar una decisión.
        """
    )
