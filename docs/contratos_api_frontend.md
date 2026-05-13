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
- **Predicción ML**: mostrar como apoyo educativo o señal indicativa. Añadir una aclaración visible del tipo: *"Este modelo usa datos sintéticos y no constituye asesoramiento financiero."*
- **`probability` baja**: informar al usuario que una probabilidad baja indica menor confianza del modelo, no un error.
- **Timeouts**: el endpoint `/market/bundle` puede tardar varios segundos si yfinance está lento. Usar un spinner o indicador de carga en la UI.
