from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# =========================================================
# Paleta y helpers de estilo
# =========================================================
ASSET_COLORS = {
    "3382.T": "#8FD3FF",
    "ATD.TO": "#1E90FF",
    "FEMSAUBD.MX": "#F6C1C1",
    "BP.L": "#FF4D4D",
    "CA.PA": "#7CFC9A",
    "ACWI": "#3FA7FF",
    "Portafolio": "#A8D8FF",
    "Benchmark": "#008CFF",
}

NORMALIZED_PRICE_COLORS = {
    "3382.T": "#2563EB",
    "ATD.TO": "#D97706",
    "FEMSAUBD.MX": "#0F766E",
    "BP.L": "#EF4444",
    "CA.PA": "#7C3AED",
    "ACWI": "#334155",
}

NORMALIZED_PRICE_FALLBACK_COLORS = [
    "#F59E0B",
    "#0891B2",
    "#4B5563",
]

METHOD_COLORS = {
    "Histórico": "#3FA7FF",
    "Paramétrico": "#FFB347",
    "Monte Carlo": "#FF6B6B",
}

INDICATOR_COLORS = {
    "Close": "#8FD3FF",
    "SMA": "#1E90FF",
    "EMA": "#F6C1C1",
    "BB_up": "#8FD3FF",
    "BB_mid": "#F6C1C1",
    "BB_low": "#FF4D4D",
    "RSI": "#8FD3FF",
    "MACD": "#8FD3FF",
    "MACD_signal": "#1E90FF",
    "MACD_hist": "#F6C1C1",
    "%K": "#8FD3FF",
    "%D": "#1E90FF",
    "Forecast": "#8FD3FF",
    "Regresión": "#1E90FF",
    "Observaciones": "#8FD3FF",
}


def _get_asset_color(name: str) -> str:
    return ASSET_COLORS.get(name, "#8FD3FF")


def _get_normalized_price_color(name: str, index: int) -> str:
    return NORMALIZED_PRICE_COLORS.get(
        name,
        NORMALIZED_PRICE_FALLBACK_COLORS[index % len(NORMALIZED_PRICE_FALLBACK_COLORS)],
    )


def _apply_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "Fecha",
    yaxis_title: str = "",
    theme: str = "dark",
) -> go.Figure:
    is_light = theme == "light"
    template = "plotly_white" if is_light else "plotly_dark"
    text_color = "#0f172a" if is_light else None
    grid_color = "rgba(15, 23, 42, 0.08)" if is_light else "rgba(255,255,255,0.10)"
    plot_bg = "#ffffff" if is_light else "rgba(0,0,0,0)"
    paper_bg = "#ffffff" if is_light else "rgba(0,0,0,0)"
    legend_bg = "rgba(255,255,255,0.96)" if is_light else "rgba(0,0,0,0)"
    legend_border = "rgba(37, 99, 235, 0.18)" if is_light else "rgba(0,0,0,0)"

    fig.update_layout(
        title=title,
        template=template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color, size=13) if is_light else None,
        title_font=dict(color=text_color, size=18) if is_light else None,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=500 if is_light else 460,
        margin=dict(l=56, r=34, t=108 if is_light else 60, b=54),
        legend=dict(
            orientation="h" if is_light else "v",
            yanchor="bottom" if is_light else "top",
            y=1.06 if is_light else 1.0,
            xanchor="center" if is_light else "left",
            x=0.5 if is_light else 1.02,
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1 if is_light else 0,
            font=dict(color=text_color, size=12) if is_light else None,
            itemsizing="constant",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=is_light,
        gridcolor=grid_color,
        zeroline=False,
        linecolor="rgba(15, 23, 42, 0.18)" if is_light else None,
        tickfont=dict(color=text_color) if is_light else None,
        title_font=dict(color=text_color) if is_light else None,
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zeroline=False,
        linecolor="rgba(15, 23, 42, 0.18)" if is_light else None,
        tickfont=dict(color=text_color) if is_light else None,
        title_font=dict(color=text_color) if is_light else None,
    )
    return fig


def _add_reference_line(fig: go.Figure, y: float, text: str = "", color: str = "rgba(255,255,255,0.35)", dash: str = "dash"):
    fig.add_hline(y=y, line_dash=dash, line_color=color)
    return fig


# =========================================================
# Módulo 0 / App
# =========================================================
def plot_normalized_prices(close: pd.DataFrame) -> go.Figure:
    base = close / close.dropna().iloc[0] * 100
    fig = go.Figure()

    for index, col in enumerate(base.columns):
        fig.add_trace(
            go.Scatter(
                x=base.index,
                y=base[col],
                mode="lines",
                name=col,
                line=dict(color=_get_normalized_price_color(col, index), width=2.9),
                opacity=0.96,
            )
        )

    return _apply_layout(fig, "Precios normalizados (base 100)", "Fecha", "Base 100", theme="light")


# =========================================================
# Módulo 1 - Técnico
# =========================================================
def plot_price_and_mas(df: pd.DataFrame, sma_col: str, ema_col: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"], mode="lines", name="Close",
            line=dict(color=INDICATOR_COLORS["Close"], width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[sma_col], mode="lines", name=sma_col,
            line=dict(color=INDICATOR_COLORS["SMA"], width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[ema_col], mode="lines", name=ema_col,
            line=dict(color=INDICATOR_COLORS["EMA"], width=2)
        )
    )

    return _apply_layout(fig, "Precio con medias móviles", "Fecha", "Precio")


def plot_bollinger(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"], mode="lines", name="Close",
            line=dict(color=INDICATOR_COLORS["Close"], width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["BB_up"], mode="lines", name="BB_up",
            line=dict(color=INDICATOR_COLORS["BB_up"], width=1.5)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["BB_mid"], mode="lines", name="BB_mid",
            line=dict(color=INDICATOR_COLORS["BB_mid"], width=1.5)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["BB_low"], mode="lines", name="BB_low",
            line=dict(color=INDICATOR_COLORS["BB_low"], width=1.5)
        )
    )

    return _apply_layout(fig, "Bandas de Bollinger", "Fecha", "Precio")


def plot_rsi(df: pd.DataFrame, rsi_col: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[rsi_col], mode="lines", name=rsi_col,
            line=dict(color=INDICATOR_COLORS["RSI"], width=2)
        )
    )

    _add_reference_line(fig, 70)
    _add_reference_line(fig, 30)

    fig.update_yaxes(range=[0, 100])

    return _apply_layout(fig, "RSI", "Fecha", "RSI")


def plot_macd(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD"], mode="lines", name="MACD",
            line=dict(color=INDICATOR_COLORS["MACD"], width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["MACD_signal"], mode="lines", name="MACD_signal",
            line=dict(color=INDICATOR_COLORS["MACD_signal"], width=2)
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["MACD_hist"], name="MACD_hist",
            marker_color=INDICATOR_COLORS["MACD_hist"],
            opacity=0.55
        )
    )

    return _apply_layout(fig, "MACD", "Fecha", "Valor")


def plot_stochastic(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["%K"], mode="lines", name="%K",
            line=dict(color=INDICATOR_COLORS["%K"], width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["%D"], mode="lines", name="%D",
            line=dict(color=INDICATOR_COLORS["%D"], width=2)
        )
    )

    _add_reference_line(fig, 80)
    _add_reference_line(fig, 20)

    fig.update_yaxes(range=[0, 100])

    return _apply_layout(fig, "Oscilador estocástico", "Fecha", "Nivel")


# =========================================================
# Módulo 2 - Rendimientos
# =========================================================
def plot_histogram_with_normal(returns: pd.Series) -> go.Figure:
    r = returns.dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    x = np.linspace(r.min(), r.max(), 300)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=r,
            histnorm="probability density",
            name="Histograma",
            nbinsx=40,
            marker_color="#8FD3FF",
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name="Normal teórica",
            line=dict(color="#008CFF", width=2)
        )
    )

    return _apply_layout(fig, "Histograma con curva normal", "Rendimiento", "Densidad")


def plot_qq(qq_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=qq_df["theoretical_quantiles"],
            y=qq_df["sample_quantiles"],
            mode="markers",
            name="Q-Q",
            marker=dict(color="#8FD3FF", size=5, opacity=0.8),
        )
    )

    min_q = float(np.nanmin([qq_df["theoretical_quantiles"].min(), qq_df["sample_quantiles"].min()]))
    max_q = float(np.nanmax([qq_df["theoretical_quantiles"].max(), qq_df["sample_quantiles"].max()]))

    fig.add_trace(
        go.Scatter(
            x=[min_q, max_q],
            y=[min_q, max_q],
            mode="lines",
            name="45°",
            line=dict(color="#008CFF", width=2),
        )
    )

    return _apply_layout(fig, "Q-Q plot", "Cuantiles teóricos", "Cuantiles muestrales")


def plot_box(returns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=returns.dropna(),
            name="Rendimientos",
            fillcolor="rgba(219, 234, 254, 0.70)",
            marker=dict(color="#2563EB", size=5, opacity=0.65),
            line=dict(color="#1E3A8A", width=2),
            boxmean=True,
        )
    )
    return _apply_layout(fig, "Boxplot de rendimientos", "", "Rendimiento")


# =========================================================
# Módulo 3 - GARCH
# =========================================================
def plot_volatility(vol_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    model_colors = {
        "ARCH(1)": "#1E3A8A",
        "GARCH(1,1)": "#7C3AED",
        "EGARCH(1,1)": "#D97706",
    }
    fallback_palette = ["#7C3AED", "#64748B", "#F59E0B"]

    for i, col in enumerate(vol_df.columns):
        color = model_colors.get(col, fallback_palette[i % len(fallback_palette)])
        fig.add_trace(
            go.Scatter(
                x=vol_df.index,
                y=vol_df[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=2.2),
            )
        )

    fig = _apply_layout(fig, "Volatilidad condicional estimada", "Fecha", "Volatilidad")
    fig.update_yaxes(rangemode="tozero")
    return fig


def plot_standardized_residuals(std_resid: pd.DataFrame | pd.Series) -> go.Figure:
    if isinstance(std_resid, pd.DataFrame):
        resid = std_resid.iloc[:, 0]
    else:
        resid = std_resid

    resid = pd.to_numeric(resid, errors="coerce").dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=resid.index,
            y=resid,
            mode="lines",
            name="Residuos estandarizados",
            line=dict(color="#2563EB", width=1.8),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(15, 23, 42, 0.35)")

    return _apply_layout(
        fig,
        "Residuos estandarizados del modelo seleccionado",
        "Fecha",
        "Residuo estandarizado",
        theme="light",
    )


def plot_forecast(forecast_df: pd.DataFrame, long_run_vol: float | None = None) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_df["horizonte"],
            y=forecast_df["volatilidad_pronosticada"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#2563EB", width=2.4),
            marker=dict(size=7, color="#2563EB"),
        )
    )

    if long_run_vol is not None:
        fig.add_hline(
            y=long_run_vol,
            line_dash="dash",
            line_color="#D97706",
            annotation_text="Volatilidad de largo plazo",
            annotation_position="top left",
        )

    fig = _apply_layout(fig, "Pronóstico de volatilidad", "Horizonte", "Volatilidad")
    fig.update_yaxes(rangemode="tozero")
    return fig


# =========================================================
# Módulo 4 - CAPM
# =========================================================
def plot_scatter_regression(x: np.ndarray, y: np.ndarray, yhat: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observaciones",
            marker=dict(color=INDICATOR_COLORS["Observaciones"], size=5, opacity=0.75),
        )
    )

    sort_idx = np.argsort(x)
    fig.add_trace(
        go.Scatter(
            x=x[sort_idx],
            y=yhat[sort_idx],
            mode="lines",
            name="Regresión",
            line=dict(color=INDICATOR_COLORS["Regresión"], width=2),
        )
    )

    return _apply_layout(fig, title, "Exceso benchmark", "Exceso activo")


# =========================================================
# Módulo 5 - VaR / CVaR
# =========================================================
def plot_var_distribution(returns: pd.Series, table: pd.DataFrame) -> go.Figure:
    r = returns.dropna()
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=r,
            nbinsx=50,
            name="Rendimientos",
            marker_color="#8FD3FF",
            opacity=0.80,
        )
    )

    for _, row in table.iterrows():
        metodo = row["método"]
        color = METHOD_COLORS.get(metodo, "#FFFFFF")

        fig.add_trace(
            go.Scatter(
                x=[-row["VaR_diario"], -row["VaR_diario"]],
                y=[0, 1],
                mode="lines",
                name=f"VaR {metodo}",
                line=dict(color=color, width=2, dash="dash"),
                yaxis="y2",
                hovertemplate=f"VaR {metodo}: %{{x:.2%}}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[-row["CVaR_diario"], -row["CVaR_diario"]],
                y=[0, 1],
                mode="lines",
                name=f"CVaR {metodo}",
                line=dict(color=color, width=2, dash="dot"),
                yaxis="y2",
                hovertemplate=f"CVaR {metodo}: %{{x:.2%}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Distribución de rendimientos con VaR y CVaR",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Rendimiento",
        yaxis_title="Frecuencia",
        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        height=460,
        margin=dict(l=40, r=120, t=60, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        barmode="overlay",
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)", zeroline=False)

    return fig


# =========================================================
# Módulo 6 - Markowitz
# =========================================================
def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de correlación",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_colorbar=dict(
            title="Correlación",
            x=1.02,
            y=0.5,
            len=0.8,
            thickness=14,
        ),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

def plot_frontier(
    sim_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    min_var: pd.Series,
    max_sharpe: pd.Series,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sim_df["volatility"],
            y=sim_df["return"],
            mode="markers",
            marker=dict(
                size=4,
                color=sim_df["sharpe"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(
                    title="Sharpe",
                    x=1.12,
                    y=0.5,
                    len=0.75,
                    thickness=16,
                ),
                opacity=0.75,
            ),
            name="Portafolios",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["return"],
            mode="lines",
            name="Frontera eficiente",
            line=dict(color="#8FD3FF", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_var["volatility"]],
            y=[min_var["return"]],
            mode="markers",
            marker=dict(size=12, symbol="diamond", color="#F6C1C1"),
            name="Mínima varianza",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[max_sharpe["volatility"]],
            y=[max_sharpe["return"]],
            mode="markers",
            marker=dict(size=14, symbol="star", color="#FF4D4D"),
            name="Máximo Sharpe",
        )
    )

    fig.update_layout(
        title="Frontera eficiente de Markowitz",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Volatilidad",
        yaxis_title="Retorno",
        height=460,
        margin=dict(l=40, r=110, t=60, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hovermode="closest",
    )

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)", zeroline=False)

    return _apply_layout(fig, "Frontera eficiente de Markowitz", "Volatilidad", "Retorno")


# =========================================================
# Módulo 8 - Benchmark
# =========================================================
def plot_benchmark_base100(port: pd.Series, bench: pd.Series) -> go.Figure:
    df = pd.concat([port, bench], axis=1).dropna()
    df.columns = ["Portafolio", "Benchmark"]
    base = df / df.iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base.index,
            y=base["Portafolio"],
            mode="lines",
            name="Portafolio",
            line=dict(color=ASSET_COLORS["Portafolio"], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=base.index,
            y=base["Benchmark"],
            mode="lines",
            name="Benchmark",
            line=dict(color=ASSET_COLORS["Benchmark"], width=2.5),
        )
    )

    return _apply_layout(fig, "Portafolio vs benchmark (base 100)", "Fecha", "Base 100")
