# Fase 2 — Fortalecimiento de /market/bundle

## Responsable

Esteban — Backend FastAPI, contrato API y diagnóstico de datos.

## Contexto

María migró las páginas de Streamlit para consumir datos desde MarketDataClient. Por eso el backend debía entregar una respuesta más explicativa para los activos incluidos y excluidos.

## Endpoint intervenido

POST /market/bundle

## Objetivo

Mantener compatibilidad con el frontend existente y agregar diagnósticos claros por ticker.

## Campos conservados

- ohlcv
- close
- returns
- missing_tickers
- last_available_date
- calendar_diagnostics

## Campos agregados

- included_tickers
- metadata

## Diagnóstico por ticker

`calendar_diagnostics` ahora incluye información por ticker con:

- raw_rows
- aligned_rows
- close_rows
- return_rows
- reason
- suggestion

## Caso de uso

Esto permite mostrar mensajes claros para activos como 3382.T cuando tienen datos descargados pero no suficientes retornos alineados en un horizonte de 1 año.

## Compatibilidad

No se cambiaron nombres ni estructuras de los campos ya usados por Streamlit.

## Validaciones

- `python -m compileall backend src pages tests`: exitoso.
- `pytest`: 3 pruebas pasaron.

## Limitaciones

`aligned_rows` usa una aproximación segura basada en filas no nulas de `close` y `returns`, y puede refinarse luego con pruebas específicas por ticker.

## Siguiente paso recomendado

Crear pruebas específicas para `POST /market/bundle` y luego avanzar a SQLAlchemy.
