# Fase 2 — Migración frontend-backend

## Objetivo

Esta fase busca alinear las páginas de Streamlit con la arquitectura definida para el proyecto:

```text
Streamlit → MarketDataClient → backend_client → FastAPI backend
```

El objetivo principal fue eliminar la descarga directa de datos de mercado desde las páginas frontend y centralizar el consumo de datos mediante `MarketDataClient`, dejando a Streamlit como capa de presentación e interacción.

---

## 1. Problema detectado

Durante la revisión se identificó una arquitectura híbrida:

- Algunas páginas ya consumían datos mediante `MarketDataClient`.
- Otras páginas seguían usando `download_single_ticker`.
- También existían referencias residuales a `yfinance`.

Esto generaba duplicidad de responsabilidades, posibles inconsistencias entre módulos y mayor dificultad para manejar errores de datos desde un único punto del sistema.

---

## 2. Archivos migrados

Los archivos frontend migrados en esta fase fueron:

- `pages/01_tecnico.py`
- `pages/02_rendimientos.py`
- `pages/03_garch.py`
- `pages/07_senales.py`

---

## 3. Cambios realizados

Los cambios aplicados fueron conservadores y enfocados únicamente en la fuente de datos:

- Se eliminó `download_single_ticker` de las páginas migradas.
- Se eliminaron referencias residuales a `yfinance` en `pages/07_senales.py`.
- Se agregó `MarketDataClient` como cliente único para consultar datos de mercado desde Streamlit.
- La obtención de datos ahora pasa por el backend.
- Se mantuvo intacta la lógica de indicadores técnicos, rendimientos, GARCH, señales, gráficos, KPIs e interpretaciones.
- En `pages/07_senales.py` se implementó una sola llamada por lote para todos los tickers del portafolio.

---

## 4. Arquitectura final

La arquitectura frontend-backend queda alineada de la siguiente forma:

```text
Streamlit
  ↓
MarketDataClient
  ↓
backend_client.py
  ↓
FastAPI backend
  ↓
Servicios de datos financieros
```

---

## 5. Validaciones realizadas

Se realizaron las siguientes validaciones:

- Búsqueda global de `download_single_ticker` en las 4 páginas migradas: sin resultados.
- Búsqueda global de `yfinance` / `yf.download` en las 4 páginas migradas: sin resultados.
- Búsqueda global de `MarketDataClient`: presente en las 4 páginas migradas.
- Compilación de las páginas migradas con:

```bash
python -m py_compile pages/01_tecnico.py pages/02_rendimientos.py pages/03_garch.py pages/07_senales.py
```

También se contempló la validación manual en Streamlit de:

- Mód.1 Análisis técnico.
- Mód.2 Rendimientos.
- Mód.3 GARCH.
- Mód.7 Señales.

---

## 6. Pendientes para backend

Queda pendiente que Esteban fortalezca el endpoint `/market/bundle` para mejorar el diagnóstico de datos. Se recomienda incluir:

- `missing_tickers` más explicativos.
- `calendar_diagnostics`.
- `reason` por ticker.
- Sugerencias para casos como `3382.T`.
- Validación de horizonte de 1 año vs 2 años.
- Mensajes de error más estructurados.

---

## 7. Riesgos controlados

La migración se realizó controlando el alcance técnico:

- No se modificó la lógica financiera.
- No se rediseñó la interfaz.
- No se tocaron `backend/main.py`, `src/api/backend_client.py` ni `src/services/market_data_client.py`.
- La migración fue conservadora y por archivo.

---

## 8. Conclusión

La parte frontend de la Fase 2 queda cerrada. Las páginas migradas ya consumen datos de mercado mediante `MarketDataClient` y backend FastAPI, sin descargas directas desde Streamlit.

El siguiente paso corresponde al backend: fortalecer los diagnósticos de datos, mejorar los mensajes por ticker y hacer más explícitas las causas de ausencia de información en casos como `3382.T`.
