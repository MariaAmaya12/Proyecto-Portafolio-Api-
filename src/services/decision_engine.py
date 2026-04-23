from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import compute_all_indicators
from src.signals import evaluate_signals


class DecisionEngine:
    """Reglas de clasificacion usadas por el panel de decision."""

    @staticmethod
    def signal_bucket(recommendation: str) -> str:
        txt = str(recommendation).lower()
        if "compra" in txt:
            return "favorable"
        if "venta" in txt:
            return "desfavorable"
        return "neutral"

    def build_signal_summary(self, ohlcv_by_ticker: dict[str, pd.DataFrame]) -> dict:
        """Resume senales tecnicas de los activos disponibles."""
        favorables = 0
        neutrales = 0
        desfavorables = 0

        for df in ohlcv_by_ticker.values():
            try:
                if df is None or df.empty:
                    continue
                signal = evaluate_signals(compute_all_indicators(df))
                if not signal:
                    continue
                bucket = self.signal_bucket(signal.get("recommendation", ""))
                if bucket == "favorable":
                    favorables += 1
                elif bucket == "desfavorable":
                    desfavorables += 1
                else:
                    neutrales += 1
            except Exception:
                continue

        if favorables > desfavorables:
            return {"lectura": "Favorable", "score": 1, "ui": "positive", "favorables": favorables, "neutrales": neutrales, "desfavorables": desfavorables}
        if desfavorables > favorables:
            return {"lectura": "Desfavorable", "score": -1, "ui": "danger", "favorables": favorables, "neutrales": neutrales, "desfavorables": desfavorables}
        return {"lectura": "Neutral", "score": 0, "ui": "warning", "favorables": favorables, "neutrales": neutrales, "desfavorables": desfavorables}

    @staticmethod
    def classify_risk(var_hist: float, persistencia: float, max_dd: float) -> dict:
        """Clasifica riesgo agregado por VaR, persistencia y drawdown."""
        dd_abs = abs(max_dd) if pd.notna(max_dd) else np.nan
        puntos = 0
        if pd.notna(var_hist):
            puntos += 2 if var_hist >= 0.03 else 1 if var_hist >= 0.015 else 0
        if pd.notna(persistencia):
            puntos += 2 if persistencia >= 0.98 else 1 if persistencia >= 0.90 else 0
        if pd.notna(dd_abs):
            puntos += 2 if dd_abs >= 0.25 else 1 if dd_abs >= 0.10 else 0

        if puntos >= 5:
            return {"nivel": "Alto", "score": -1, "ui": "danger", "mensaje": "El portafolio presenta perdidas extremas potenciales y/o persistencia de volatilidad suficientemente elevadas como para justificar cautela."}
        if puntos >= 3:
            return {"nivel": "Medio", "score": 0, "ui": "warning", "mensaje": "El portafolio muestra un riesgo intermedio: no obliga a salir completamente, pero si a controlar tamano de posicion y exposicion."}
        return {"nivel": "Bajo", "score": 1, "ui": "positive", "mensaje": "El perfil de riesgo luce relativamente contenido para la ventana analizada."}

    @staticmethod
    def classify_benchmark(summary_df: pd.DataFrame, extras_df: pd.DataFrame) -> dict:
        """Clasifica desempeno relativo contra benchmark."""
        if summary_df.empty:
            return {"nivel": "No concluyente", "score": 0, "ui": "warning", "mensaje": "No fue posible construir una comparacion robusta frente al benchmark."}
        try:
            port_ret = float(summary_df.loc[summary_df["serie"] == "Portafolio", "ret_anualizado"].iloc[0])
            bench_ret = float(summary_df.loc[summary_df["serie"] == "Benchmark", "ret_anualizado"].iloc[0])
            alpha = np.nan
            if not extras_df.empty and "Alpha de Jensen" in extras_df["metrica"].values:
                alpha = float(extras_df.loc[extras_df["metrica"] == "Alpha de Jensen", "valor"].iloc[0])
            if port_ret > bench_ret and (pd.isna(alpha) or alpha >= 0):
                return {"nivel": "Superior", "score": 1, "ui": "positive", "mensaje": "El portafolio presenta una lectura relativa favorable frente al benchmark en la ventana analizada."}
            if port_ret < bench_ret and (pd.isna(alpha) or alpha < 0):
                return {"nivel": "Inferior", "score": -1, "ui": "danger", "mensaje": "El portafolio viene rezagado frente al benchmark y no muestra una superioridad relativa consistente."}
            return {"nivel": "No concluyente", "score": 0, "ui": "warning", "mensaje": "La comparacion relativa no muestra una ventaja consistente; algunas metricas son mixtas y no confirman dominancia clara."}
        except Exception:
            return {"nivel": "No concluyente", "score": 0, "ui": "warning", "mensaje": "No fue posible construir una lectura comparativa suficientemente robusta."}

    @staticmethod
    def final_decision(risk_score: int, signal_score: int, bench_score: int) -> dict:
        """Combina scores parciales en una postura final."""
        total = risk_score + signal_score + bench_score
        if total >= 2:
            return {"titulo": "Compra tactica", "ui": "positive", "mensaje_general": "La lectura integrada favorece una postura compradora o de incremento tactico de exposicion.", "mensaje_riesgo": "El principal riesgo es que un cambio brusco de mercado revierta la senal tecnica y deteriore el perfil de volatilidad.", "mensaje_formal": "La combinacion de riesgo contenido, sesgo tecnico favorable y comparacion relativa no adversa respalda una postura de compra tactica dentro de la ventana analizada.", "score_total": total}
        if total >= 0:
            return {"titulo": "Mantener / compra selectiva", "ui": "warning", "mensaje_general": "La lectura integrada permite mantener exposicion y considerar compras selectivas, pero no justifica una expansion agresiva de riesgo.", "mensaje_riesgo": "El principal riesgo es entrar con confirmacion incompleta y enfrentar un deterioro posterior en benchmark o volatilidad.", "mensaje_formal": "La evidencia agregada no es suficientemente fuerte para una postura agresiva, pero tampoco justifica deshacer exposicion. La decision razonable es mantener y, en todo caso, comprar de forma selectiva.", "score_total": total}
        if total == -1:
            return {"titulo": "Reducir exposicion", "ui": "warning", "mensaje_general": "La lectura integrada sugiere reducir parcialmente exposicion o evitar nuevas compras hasta que mejoren las condiciones.", "mensaje_riesgo": "El riesgo central es mantener una posicion relativamente alta en un contexto donde la evidencia agregada se ha debilitado.", "mensaje_formal": "La combinacion de senales no favorece una ampliacion de posicion. La decision mas consistente es reducir exposicion marginalmente y esperar mejor confirmacion estadistica y tecnica.", "score_total": total}
        return {"titulo": "Venta / postura defensiva", "ui": "danger", "mensaje_general": "La lectura integrada favorece una postura defensiva: reducir exposicion de forma relevante o priorizar salida.", "mensaje_riesgo": "El principal riesgo es permanecer sobreexpuesto en un entorno donde coinciden riesgo elevado, deterioro tecnico y/o rezago relativo.", "mensaje_formal": "La evidencia integrada es adversa. Desde una perspectiva de control de riesgo, la postura mas defendible es de venta o reduccion sustancial de exposicion.", "score_total": total}

