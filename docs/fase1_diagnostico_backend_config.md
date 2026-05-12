# Fase 1 — Diagnóstico técnico de backend y configuración

## Responsable

Esteban — Backend, configuración, arquitectura, pruebas, SQLAlchemy, Docker y CI.

## Objetivo

Validar el estado técnico inicial del backend, configuración del proyecto, dependencias, archivos sensibles, pruebas y preparación para las siguientes fases.

## Estado general del backend

El proyecto ya cuenta con un backend FastAPI ubicado en `backend/main.py`. Este backend expone endpoints funcionales para los módulos financieros existentes, incluyendo datos de mercado, snapshot macroeconómico, rendimientos, indicadores técnicos, señales, VaR/CVaR, Markowitz y CAPM.

Cuando el backend está corriendo, la documentación Swagger puede consultarse en:

```text
http://127.0.0.1:8000/docs
```

## Endpoints actuales

| Método | Ruta | Propósito | Estado |
| --- | --- | --- | --- |
| GET | `/health` | Verificar disponibilidad del backend. | Funcional |
| GET | `/` | Verificar respuesta raíz del servicio. | Funcional |
| HEAD | `/` | Verificar disponibilidad raíz sin cuerpo de respuesta. | Funcional |
| GET | `/macro/snapshot` | Consultar snapshot macroeconómico. | Funcional |
| POST | `/market/bundle` | Obtener OHLCV, matriz de cierres y retornos para varios tickers. | Funcional |
| GET | `/returns/{ticker}` | Calcular retornos, estadísticos descriptivos y pruebas de normalidad. | Funcional |
| GET | `/indicators/{ticker}` | Calcular indicadores técnicos por activo. | Funcional |
| POST | `/signals/evaluate` | Evaluar señales técnicas para un activo. | Funcional |
| POST | `/risk/var-cvar` | Calcular métricas VaR y CVaR. | Funcional |
| POST | `/portfolio/markowitz` | Ejecutar optimización de portafolio y frontera eficiente. | Funcional |
| GET | `/capm/{ticker}` | Calcular métricas CAPM para un activo frente a benchmark. | Funcional |

## Configuración del proyecto

- `.gitignore`: existe y contiene reglas para evitar subir entornos virtuales, variables sensibles, caches, datos generados, bases locales, modelos generados, cobertura, reportes e información de IDE.
- `.env.example`: existe y sirve como plantilla de configuración local sin exponer secretos reales.
- `requirements.txt`: existe y concentra las dependencias actuales del proyecto.
- Estructura principal: existen las carpetas `backend/`, `pages/`, `src/` y `tests/`.
- `docs/`: no existía inicialmente antes de crear este archivo de diagnóstico técnico.

## Archivos sensibles o generados

No deben subirse al repositorio:

- `.env`
- `.env.local`
- `.venv/`
- `venv/`
- `__pycache__/`
- `.pytest_cache/`
- `data/yfinance_cache/`
- `data/raw/`
- `data/processed/`
- `*.sqlite`
- `*.db`
- `models/*.joblib`
- `models/*.pkl`

## Dependencias

`requirements.txt` existe, pero debe revisarse con cuidado antes de agregar dependencias nuevas. Cada dependencia adicional debe tener una justificación clara para evitar aumentar innecesariamente el peso del entorno y el riesgo de incompatibilidades.

Dependencias previstas para fases futuras:

- SQLAlchemy
- httpx
- scikit-learn
- joblib
- pydantic-settings, si se adopta configuración avanzada

## Pruebas

Comandos de validación local:

```bash
pytest
uvicorn backend.main:app --reload
```

También se debe revisar:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

Validación ejecutada en Fase 1:

- `git status --short`: sin cambios pendientes.
- `pytest`: 3 pruebas ejecutadas correctamente.
- `python -m pytest`: falló por diferencia de entorno/intérprete, no por fallo del código.
- `python -m compileall backend src tests`: sin errores de sintaxis.
- Observación: el ejecutable `pytest` disponible en terminal usa un entorno donde pytest existe, pero `/usr/local/bin/python` no tiene instalado el módulo pytest. Se recomienda alinear el entorno antes de CI/Docker.

## Riesgos detectados

- Arquitectura híbrida frontend-backend.
- Falta de SQLAlchemy.
- Falta de ML.
- Falta de Dockerfile.
- Falta de `docker-compose.yml`.
- Falta de CI general de pruebas.
- Falta de endpoints de renta fija.
- Falta de endpoints de opciones.
- Falta de endpoints de stress testing.
- Posible falta de tags en Swagger.

## Recomendaciones para Fase 2

- Fortalecer `POST /market/bundle`.
- Agregar diagnósticos por ticker.
- Reportar activos incluidos y excluidos.
- Explicar casos como `3382.T` cuando no tiene datos suficientes o retornos alineados.
- Apoyar la migración progresiva de Streamlit hacia consumo completo del backend.

## Recomendaciones para Fase 3

- Modularizar `backend/main.py` en routers.
- Agregar tags de Swagger.
- Crear capa SQLAlchemy.
- Agregar endpoint `GET /db/health`.
- Separar `schemas`, `routes`, `db` y `services`.

## Checklist de validación local

- [ ] pytest ejecutado
- [ ] backend inicia con uvicorn
- [ ] /docs abre correctamente
- [ ] /health responde correctamente
- [ ] no se modificaron archivos de frontend
- [ ] no se mezclaron cambios de María
