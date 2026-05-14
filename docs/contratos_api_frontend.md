# Contratos API para Frontend

## Objetivo
Documentar los endpoints backend disponibles para que Streamlit pueda consumirlos de forma estable, sin necesidad de modificar el backend.

---

## Base URL

**Desarrollo local:**
```
http://127.0.0.1:8000
```

**Docker Compose (red interna):**
```
http://backend:8000
```

**Variables de entorno soportadas por el cliente frontend (`src/api/backend_client.py`):**
- `API_BASE_URL`
- `BACKEND_API_BASE_URL`

El cliente intenta resolver la URL en ese orden. Si ninguna está definida, usa `http://127.0.0.1:8000` como fallback.

---

## GET /health

**Propósito:** Verificar que FastAPI está disponible y respondiendo.

**Response esperado (`200`):**
```json
{
  "status": "ok"
}
```

---

## GET /db/health

**Propósito:** Verificar la conexión con la base de datos SQLite a través de SQLAlchemy.

**Response esperado (`200`):**
```json
{
  "status": "ok",
  "database": "sqlite"
}
```

**Error posible (`503`):** si la base de datos no está accesible.

---

## POST /market/bundle

**Propósito:** Obtener datos históricos OHLCV, precios de cierre, retornos y diagnósticos por ticker.

**Request:**
```json
{
  "tickers": ["AAPL", "3382.T"],
  "start": "2024-01-01",
  "end": "2024-12-31"
}
```

**Campos principales del response (`200`):**

| Campo | Descripción |
|---|---|
| `ohlcv` | Datos OHLCV por ticker |
| `close` | Serie de precios de cierre alineados |
| `returns` | Serie de retornos diarios alineados |
| `included_tickers` | Activos con datos suficientes para análisis |
| `missing_tickers` | Activos excluidos por falta de datos |
| `last_available_date` | Último día con datos disponibles |
| `calendar_diagnostics` | Diagnóstico general de alineación de fechas |
| `metadata` | Información general de la consulta |

**Campos clave para el frontend:**

- **`included_tickers`**: lista de activos que tienen datos suficientes en el rango solicitado. El frontend debe operar solo con estos activos.
- **`missing_tickers`**: lista de activos excluidos. El frontend debe informar al usuario cuáles y por qué.
- **`calendar_diagnostics.by_ticker`**: diagnóstico individual por activo, con motivo de exclusión y sugerencia.
- **`metadata`**: fuente de datos, rango efectivo y marca de tiempo de generación.

**Ejemplo de `calendar_diagnostics.by_ticker`:**
```json
{
  "3382.T": {
    "raw_rows": 245,
    "aligned_rows": 0,
    "close_rows": 245,
    "return_rows": 0,
    "reason": "insufficient aligned returns",
    "suggestion": "Use a longer horizon, for example 2 years, or analyze individually"
  }
}
```

**Errores posibles:**
- `404` si ningún ticker tiene datos en el rango solicitado.
- `422` si el payload es inválido (tickers vacíos, fechas incorrectas, `end < start`).
- `502` si el proveedor externo (yfinance) no responde.
- `503` si el servicio no puede procesar la solicitud.

---

## POST /ml/predict

**Propósito:** Obtener una predicción de señal de mercado (Alcista / Bajista / Neutral) a partir de indicadores técnicos simples.

**Request:**
```json
{
  "close": 110.0,
  "sma": 100.0,
  "ema": 105.0,
  "rsi": 55.0
}
```

| Campo | Tipo | Restricciones |
|---|---|---|
| `close` | float | > 0 |
| `sma` | float | > 0 |
| `ema` | float | > 0 |
| `rsi` | float | 0 ≤ rsi ≤ 100 |

**Response esperado (`200`):**
```json
{
  "prediction": "Alcista",
  "probability": 0.87,
  "model_version": "v1",
  "log_id": 1
}
```

| Campo | Descripción |
|---|---|
| `prediction` | Señal predicha: `"Alcista"`, `"Bajista"` o `"Neutral"` |
| `probability` | Confianza del modelo en la predicción (0.0 – 1.0) |
| `model_version` | Versión del modelo desplegado (actualmente `"v1"`) |
| `log_id` | ID del registro en base de datos (referencia técnica) |

**Notas para el frontend:**
- `prediction` siempre será uno de `{"Alcista", "Bajista", "Neutral"}`.
- `probability` puede ser baja para la clase `"Alcista"` (clase minoritaria en el entrenamiento); esto es esperado y no indica error.
- `log_id` es solo una referencia técnica interna; no necesita mostrarse al usuario final.
- **El modelo usa datos sintéticos. La predicción no debe presentarse como recomendación financiera real.** Sugerencia de label en UI: *"Señal estimada (educativa)"*.

**Errores posibles:**
- `422` si algún campo tiene valor inválido (`rsi=150`, `close="texto"`, etc.).
- `503` si el modelo `.joblib` no está disponible o falla la persistencia en base de datos.

---

## POST /ml/risk-score

**Propósito:** Estimar un score auxiliar de riesgo financiero a 5 días usando Machine Learning.

**Request:**
```json
{
  "ret_1d": -0.005,
  "ret_5d": -0.018,
  "ret_20d": 0.012,
  "vol_5d": 0.014,
  "vol_20d": 0.011,
  "rsi": 42.0,
  "macd_hist": -0.003,
  "bb_position": 0.28,
  "close_over_sma20": 0.975,
  "drawdown_20d": -0.031
}
```

| Campo | Tipo | Descripción |
|---|---|---|
| `ret_1d` | float | Retorno diario simple |
| `ret_5d` | float | Retorno acumulado a 5 días |
| `ret_20d` | float | Retorno acumulado a 20 días |
| `vol_5d` | float | Volatilidad de retornos a 5 días (≥ 0) |
| `vol_20d` | float | Volatilidad de retornos a 20 días (≥ 0) |
| `rsi` | float | RSI 14 (0 ≤ rsi ≤ 100) |
| `macd_hist` | float | Histograma MACD normalizado por precio |
| `bb_position` | float | Posición en Bandas de Bollinger (-1 ≤ bb_position ≤ 2) |
| `close_over_sma20` | float | Distancia relativa al SMA 20 |
| `drawdown_20d` | float | Caída desde máximo de 20 días (≤ 0) |

**Response esperado (`200`):**
```json
{
  "risk_score": 0.73,
  "risk_level": "alto",
  "horizon_days": 5,
  "model_version": "risk-v1",
  "log_id": 1
}
```

| Campo | Descripción |
|---|---|
| `risk_score` | Probabilidad estimada de evento adverso (0.0 – 1.0) |
| `risk_level` | Nivel derivado: `"bajo"` (< 0.35), `"moderado"` (0.35 – 0.60), `"alto"` (> 0.60) |
| `horizon_days` | Horizonte de predicción en días (actualmente `5`) |
| `model_version` | Versión del modelo (actualmente `"risk-v1"`) |
| `log_id` | ID del registro en base de datos (referencia técnica) |

**Notas para el frontend:**
- `risk_score` siempre estará entre `0.0` y `1.0`.
- `risk_level` siempre será uno de `{"bajo", "moderado", "alto"}`.
- `horizon_days` y `model_version` son informativos; pueden cambiar en versiones futuras.
- `log_id` es solo una referencia técnica interna; no necesita mostrarse al usuario final.
- **Este score es auxiliar y educativo, no constituye una recomendación financiera.** Sugerencia de label en UI: *"Score de riesgo estimado (educativo)"*.
- Se puede mostrar como una tarjeta de riesgo ML en Streamlit, complementando VaR, CVaR y volatilidad.

**Errores posibles:**
- `422` si algún campo tiene valor inválido o está ausente.
- `503` si el modelo `.joblib` no está disponible o falla la persistencia en base de datos.

---

## Endpoints financieros

Los siguientes endpoints exponen modelos financieros analíticos para consumo desde Streamlit. Todos aceptan JSON, están disponibles en `/docs` y `/redoc`, y sus resultados son académicos/analíticos: **no constituyen recomendaciones financieras**.

**Reglas generales de error:**
- `422` si el payload no cumple las validaciones Pydantic.
- `400` si el payload es válido en forma, pero genera un error de dominio financiero.
- `500` si ocurre un error inesperado del backend.

### POST /volatility/ewma

**Propósito:** Calcular volatilidad y varianza EWMA sobre una serie de retornos.

**Request:**
```json
{
  "returns": [0.01, -0.02, 0.015, -0.005, 0.008],
  "lambda_": 0.94,
  "annualize": false,
  "periods_per_year": 252
}
```

**Response esperado (`200`):**
```json
{
  "ewma_volatility": 0.0123,
  "ewma_variance": 0.00015,
  "lambda_": 0.94,
  "annualize": false,
  "periods_per_year": 252,
  "observations": 5
}
```

**Uso frontend:**
- Módulo de volatilidad.
- Comparación histórica / EWMA / GARCH.
- Mostrar KPI de volatilidad suavizada.

### POST /fixed-income/bond-metrics

**Propósito:** Calcular precio, duración de Macaulay, duración modificada y convexidad de un bono.

**Request:**
```json
{
  "face_value": 1000.0,
  "coupon_rate": 0.05,
  "market_rate": 0.04,
  "maturity_years": 5.0,
  "frequency": 1
}
```

**Response esperado (`200`):**
```json
{
  "price": 1044.52,
  "macaulay_duration": 4.55,
  "modified_duration": 4.37,
  "convexity": 24.12
}
```

**Uso frontend:**
- Página de renta fija.
- Métricas de sensibilidad ante tasas.
- KPIs: precio, duración, duración modificada y convexidad.

### POST /fixed-income/nelson-siegel

**Propósito:** Calcular tasas estimadas de una curva Nelson-Siegel para vencimientos dados.

**Request:**
```json
{
  "maturities": [0.5, 1.0, 2.0, 5.0, 10.0],
  "beta0": 0.05,
  "beta1": -0.02,
  "beta2": 0.01,
  "tau": 1.5
}
```

**Response esperado (`200`):**
```json
{
  "maturities": [0.5, 1.0, 2.0, 5.0, 10.0],
  "yields": [0.034, 0.038, 0.042, 0.047, 0.049]
}
```

**Uso frontend:**
- Curva de tasas.
- Gráfica maturity vs yield.
- Simulación de estructura temporal.

### POST /options/black-scholes

**Propósito:** Calcular precios teóricos call y put bajo Black-Scholes.

**Request:**
```json
{
  "spot": 100.0,
  "strike": 100.0,
  "rate": 0.05,
  "volatility": 0.2,
  "time_to_maturity": 1.0
}
```

**Response esperado (`200`):**
```json
{
  "call_price": 10.45,
  "put_price": 5.57
}
```

**Uso frontend:**
- Página de opciones.
- Valorar call y put europea.
- Mostrar precio teórico bajo Black-Scholes.

### POST /options/greeks

**Propósito:** Calcular sensibilidades de opciones europeas bajo Black-Scholes.

**Request:**
```json
{
  "spot": 100.0,
  "strike": 100.0,
  "rate": 0.05,
  "volatility": 0.2,
  "time_to_maturity": 1.0
}
```

**Response esperado (`200`):**
```json
{
  "delta_call": 0.63,
  "delta_put": -0.36,
  "gamma": 0.018,
  "vega": 37.52,
  "theta_call": -6.41,
  "theta_put": -1.65,
  "rho_call": 53.23,
  "rho_put": -41.89
}
```

**Uso frontend:**
- Sensibilidad de opciones.
- KPIs o tabla de Greeks.
- Interpretación de riesgo por parámetro.

### POST /stress/portfolio

**Propósito:** Aplicar choques por activo a retornos de portafolio y resumir métricas de riesgo.

**Request:**
```json
{
  "returns": [
    [0.01, -0.005],
    [-0.02, 0.004],
    [0.015, -0.003],
    [-0.01, -0.002],
    [0.008, 0.006]
  ],
  "weights": [0.6, 0.4],
  "shocks": {
    "asset_0": -0.01,
    "asset_1": -0.005
  },
  "confidence_level": 0.95
}
```

**Response esperado (`200`):**
```json
{
  "mean_return": -0.0042,
  "min_return": -0.0184,
  "max_drawdown": -0.025,
  "var": 0.017,
  "cvar": 0.019,
  "observations": 5
}
```

**Uso frontend:**
- Página de stress testing.
- Simulación de choques por activo.
- KPIs: VaR, CVaR, drawdown máximo, retorno mínimo.

### POST /stress/combined-scenario

**Propósito:** Aplicar un escenario adverso combinado de precios, volatilidad y tasas.

**Request:**
```json
{
  "prices": [100.0, 101.0, 99.0, 98.0],
  "returns": [0.01, -0.02, 0.005, -0.01],
  "weights": 1.0,
  "price_shock": -0.1,
  "volatility_multiplier": 1.5,
  "bond_price": 1000.0,
  "modified_duration": 4.5,
  "convexity": 20.0,
  "rate_shock": 0.01
}
```

**Response esperado (`200`):**
```json
{
  "scenario_summary": {
    "mean_return": -0.0056,
    "min_return": -0.03,
    "max_drawdown": -0.04,
    "var": 0.025,
    "cvar": 0.03
  },
  "shocked_bond_price": 956.0,
  "shocked_prices": [90.0, 90.9, 89.1, 88.2],
  "shocked_returns": [0.015, -0.03, 0.0075, -0.015]
}
```

**Uso frontend:**
- Escenario adverso combinado.
- Shock de precios, volatilidad y tasas.
- Panel de interpretación del riesgo agregado.

---

## Manejo de errores esperado

Todos los errores siguen el formato estándar del backend:

```json
{
  "error": "Descripción del error.",
  "detail": [
    {
      "field": "nombre_del_campo",
      "message": "Mensaje descriptivo."
    }
  ]
}
```

| Código | Causa típica |
|---|---|
| `422` | Payload inválido (Pydantic / validación de negocio) |
| `404` | Ticker sin datos en el rango solicitado |
| `502` | Fallo del proveedor externo (yfinance) |
| `503` | Backend, modelo ML o base de datos no disponible |

---

## Recomendaciones para Streamlit

- **`missing_tickers` no vacío**: mostrar un aviso claro al usuario indicando qué activos fueron excluidos y por qué (usar `calendar_diagnostics.by_ticker[ticker].suggestion`).
- **`included_tickers`**: usar esta lista para construir cualquier análisis posterior; no asumir que todos los tickers del request están disponibles.
- **Fallo parcial**: no bloquear toda la aplicación si un ticker falla. Continuar con los activos disponibles en `included_tickers`.
- **Predicción ML (`/ml/predict`)**: mostrar como apoyo educativo o señal indicativa. Añadir una aclaración visible del tipo: *"Este modelo usa datos sintéticos y no constituye asesoramiento financiero."*
- **`probability` baja**: informar al usuario que una probabilidad baja indica menor confianza del modelo, no un error.
- **Risk score ML (`/ml/risk-score`)**: mostrar como apoyo al análisis de riesgo, no como señal de compra o venta. Combinar con métricas cuantitativas existentes (VaR, CVaR, volatilidad, drawdown) para dar contexto al usuario. Sugerencia de visualización: tarjeta con color según `risk_level` (verde = bajo, amarillo = moderado, rojo = alto) y el valor numérico de `risk_score`.
- **Timeouts**: el endpoint `/market/bundle` puede tardar varios segundos si yfinance está lento. Usar un spinner o indicador de carga en la UI.
