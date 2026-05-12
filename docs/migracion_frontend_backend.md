# Diagnóstico de migración Frontend-Backend

## Objetivo

Documentar el estado actual de integración entre Streamlit y el backend FastAPI, identificando qué módulos ya consumen la API, qué módulos deben revisarse y qué acciones se deben priorizar en la Fase 2.

La meta del proyecto debe ser que Streamlit funcione como capa de presentación e interacción, mientras que el backend centraliza la descarga, validación, limpieza, cálculo y respuesta de datos.

---

## 1. Arquitectura deseada

El flujo deseado del proyecto es:

```text
Usuario
  ↓
Streamlit
  ↓
src/services/market_data_client.py
  ↓
src/api/backend_client.py
  ↓
FastAPI backend
  ↓
Servicios financieros / proveedores externos / base de datos
```

Esto permite:

- evitar lógica duplicada,
- centralizar errores,
- mejorar mantenibilidad,
- facilitar pruebas,
- hacer más clara la sustentación,
- y cumplir mejor con la rúbrica del proyecto.

---

## 2. Arquitectura observada

Con base en los pantallazos y archivos revisados, el proyecto ya cuenta con integración backend mediante:

```text
src/api/backend_client.py
src/services/market_data_client.py
```

Estos archivos son una fortaleza del proyecto porque permiten centralizar llamadas al backend.

Sin embargo, también se identificó que algunas páginas podrían seguir usando rutas antiguas de datos, como:

```text
yfinance
yf.download
download_single_ticker
src/download.py
```

Por tanto, el proyecto se encuentra en una arquitectura híbrida.

---

## 3. Archivos clave de integración

| Archivo | Función actual | Evaluación |
|---|---|---|
| `src/api/backend_client.py` | Centraliza URL base, errores, timeouts, reintentos y conversión a DataFrame. | Bien estructurado. Debe mantenerse. |
| `src/services/market_data_client.py` | Cliente reutilizable para solicitar datos de mercado al backend. | Correcto. Debe usarse en las páginas Streamlit. |
| `src/download.py` | Utilidad de descarga directa. | Debe revisarse como posible componente legacy. |
| `pages/*.py` | Interfaz visual de módulos. | Deben consumir backend de manera homogénea. |
| `backend/main.py` o `backend/api/*` | Expone endpoints FastAPI. | Debe fortalecerse con routers, tags y endpoints faltantes. |

---

## 4. Matriz de integración por módulo

| Página / módulo | Estado actual estimado | Riesgo | Acción recomendada para Fase 2 |
|---|---|---|---|
| Inicio | Consume datos agregados, pero excluye `3382.T` en 1 año. | Medio-alto | Mostrar diagnóstico claro de tickers excluidos. |
| Contextualización | Principalmente estática/descriptiva. | Bajo | Mantener, pero conectar con decisión final. |
| Análisis técnico | Funcional, pero requiere revisar flujo de datos. | Medio | Confirmar si usa backend; si no, migrar a `/indicators/{ticker}` o `/market/bundle`. |
| Rendimientos | Funcional, requiere validar fuente. | Medio | Migrar completamente a backend si usa descarga directa. |
| GARCH | Funcional, requiere revisar fuente. | Medio | Crear o usar endpoint de volatilidad; incluir EWMA en fases futuras. |
| CAPM | Parece integrado con backend. | Bajo | Mantener y validar consistencia de datos. |
| VaR/CVaR | Parece integrado con backend. | Bajo | Mantener; agregar backtesting visible. |
| Markowitz | Parece integrado con backend. | Bajo | Mantener; validar pesos y datos alineados. |
| Señales | Funcional, pero requiere revisar si usa flujo local. | Medio | Migrar a endpoint `/signals/evaluate`. |
| Macro y benchmark | Parece integrado con backend. | Bajo | Mostrar fuente y fecha de datos. |
| Panel de decisión | Integra resultados, pero debe fortalecerse. | Medio | Conectar nuevos módulos y explicar regla de decisión. |

---

## 5. Búsquedas recomendadas en VS Code

Para validar el estado real del código, buscar con `Ctrl + Shift + F`:

```text
yfinance
yf.download
download_single_ticker
fetch_market_bundle_from_backend
MarketDataClient
backend_get
backend_post
```

Interpretación:

| Resultado encontrado | Significado |
|---|---|
| `yfinance` en páginas Streamlit | Posible descarga directa desde frontend. |
| `yf.download` | Descarga directa que debe migrarse o justificarse. |
| `download_single_ticker` | Flujo legacy o local. |
| `MarketDataClient` | Buena práctica de integración. |
| `fetch_market_bundle_from_backend` | Consumo directo del endpoint `/market/bundle`. |
| `backend_get` / `backend_post` | Uso del cliente centralizado del backend. |

---

## 6. Problema especial: `3382.T`

### Observación

Al seleccionar horizonte de 1 año, el ticker `3382.T` aparece como sin datos disponibles o se excluye del análisis conjunto. Al seleccionar 2 años, la información carga correctamente.

### Interpretación probable

El problema puede estar asociado con:

- datos insuficientes para el horizonte seleccionado,
- fechas no alineadas con los demás activos,
- diferencias de calendario bursátil,
- días festivos o zona horaria,
- exclusión por `dropna()` o validación de retornos,
- diferencia entre análisis individual y análisis conjunto.

### Acción requerida para backend

El endpoint `/market/bundle` debería devolver más información:

```json
{
  "included_tickers": ["ATD.TO", "FEMSAUBD.MX", "BP.L", "CA.PA"],
  "missing_tickers": ["3382.T"],
  "calendar_diagnostics": {
    "3382.T": {
      "raw_rows": 240,
      "aligned_rows": 0,
      "first_date": "2025-05-13",
      "last_date": "2026-05-10",
      "reason": "insufficient aligned returns",
      "suggestion": "Increase horizon to 2 years or analyze ticker individually"
    }
  }
}
```

### Acción requerida para Streamlit

Streamlit debería mostrar:

```text
3382.T fue excluido del análisis conjunto porque no tiene suficientes retornos alineados con los demás activos en el horizonte seleccionado. Puedes ampliar el horizonte a 2 años o analizarlo individualmente.
```

---

## 7. Contratos mínimos que deben quedar claros

Para que María y Esteban trabajen sin conflicto, antes de migrar páginas se deben definir contratos de API.

### `/market/bundle`

Debe devolver:

```text
ohlcv
close
returns
missing_tickers
last_available_date
calendar_diagnostics
metadata
```

### `/indicators/{ticker}`

Debe devolver:

```text
ticker
period
last_price
daily_change
rsi
macd
sma
ema
bollinger
technical_signal
```

### `/returns/{ticker}`

Debe devolver:

```text
returns
mean
volatility
skewness
kurtosis
normality_test
summary
```

### `/signals/evaluate`

Debe devolver:

```text
signal
score
buy_signals
sell_signals
neutral_signals
reasons
```

---

## 8. Reglas de arquitectura para Fase 2

1. Streamlit no debe llamar directamente a `yfinance`.
2. Streamlit no debe descargar precios directamente.
3. La descarga y limpieza de datos debe vivir en backend o servicios usados por backend.
4. Las páginas deben usar `MarketDataClient` o `backend_client.py`.
5. Los errores deben convertirse en mensajes comprensibles para el usuario.
6. Si un ticker se excluye, se debe explicar el motivo.
7. Cada página migrada debe probarse antes de pasar a la siguiente.

---

## 9. Plan de migración sugerido

Orden recomendado:

```text
1. Rendimientos
2. Análisis técnico
3. GARCH / volatilidad
4. Señales
5. Inicio / mensaje de tickers excluidos
6. Panel de decisión
```

La razón es que rendimientos y análisis técnico son la base de varios módulos posteriores. GARCH y señales dependen de datos limpios y consistentes.

---

## 10. Validaciones necesarias

Después de migrar cada página:

```bash
streamlit run app.py
uvicorn backend.main:app --reload
pytest
```

Validar manualmente:

- horizonte 1 año,
- horizonte 2 años,
- ticker `3382.T`,
- activos restantes,
- páginas que antes funcionaban,
- mensajes de error,
- tiempo de carga,
- que no aparezcan errores técnicos al usuario.

---

## 11. Conclusión

La integración frontend-backend está bien encaminada, pero todavía no debe considerarse cerrada. La Fase 2 debe enfocarse en eliminar la arquitectura híbrida, centralizar el flujo de datos y mejorar el diagnóstico de activos excluidos. Esto permitirá avanzar con mayor seguridad hacia SQLAlchemy, nuevos modelos, ML y rediseño de la app.
