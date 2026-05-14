# RiskLab USTA — Dashboard de Gestión de Riesgo Financiero con FastAPI, Streamlit e IA

## Descripción general

RiskLab USTA es un proyecto académico para análisis y gestión de riesgo financiero. Integra un dashboard interactivo en Streamlit, un backend propio en FastAPI, modelos cuantitativos de riesgo, modelos financieros avanzados, componentes de Machine Learning, persistencia con SQLite y una infraestructura declarativa con Docker y CI.

El objetivo del proyecto es centralizar la lógica financiera y analítica en servicios reutilizables, evitando que la interfaz duplique fórmulas o dependa directamente de proveedores externos. Streamlit actúa como capa visual y FastAPI como contrato estable para cálculos, datos y predicciones.

## Arquitectura

```text
Usuario
  ↓
Streamlit
  ↓
FastAPI
  ↓
src/ modelos financieros y analítica
  ↓
servicios de mercado/cache/modelos ML/SQLite
```

Responsabilidades principales:

- `app.py` y `pages/`: interfaz Streamlit, navegación, visualización de KPIs, tablas, gráficos e interpretación.
- `src/`: lógica financiera pura, analítica cuantitativa, modelos de riesgo, servicios de mercado y utilidades de UI.
- `backend/`: API FastAPI, rutas, schemas Pydantic, persistencia, carga de modelos ML y endpoints.
- `backend/api/`: rutas modulares para volatilidad, renta fija, opciones y stress testing.
- `backend/schemas/`: contratos de entrada y salida para modelos financieros avanzados.
- `backend/ml/`: entrenamiento, carga cacheada y predicción de modelos ML.
- `models/`: modelos ML serializados en `.joblib`.
- `tests/`: pruebas unitarias y pruebas de contrato API.
- `docs/`: documentación técnica por fases y contratos frontend-backend.
- `report/`: informe académico del proyecto.

## Funcionalidades principales

- Contextualización del portafolio y activos.
- Análisis técnico con indicadores.
- Rendimientos, estadística descriptiva y hechos estilizados.
- Volatilidad GARCH y EWMA.
- CAPM y métricas frente a benchmark.
- VaR, CVaR y backtesting con test de Kupiec.
- Optimización de portafolios con Markowitz.
- Evaluación de señales técnicas.
- Contexto macroeconómico y benchmark.
- Machine Learning predictivo para señales.
- Risk score predictivo de riesgo financiero.
- Renta fija: precio, duración, duración modificada y convexidad.
- Curva Nelson-Siegel.
- Valoración de opciones Black-Scholes.
- Greeks de opciones.
- Stress testing de portafolio.
- Escenario combinado de precios, volatilidad y tasas.

## API backend

La API está disponible en FastAPI y se documenta automáticamente en `/docs` y `/redoc`. No se limita a un número fijo de endpoints; está organizada por módulos.

### Salud y datos

- `GET /health`: verifica disponibilidad básica de la API.
- `GET /db/health`: verifica conexión con SQLite mediante SQLAlchemy.
- `GET /macro/snapshot`: obtiene el snapshot macroeconómico.
- `POST /market/bundle`: obtiene precios, OHLCV, retornos y diagnósticos por ticker.

### Análisis financiero base

- `GET /returns/{ticker}`: retornos, estadísticos descriptivos y pruebas asociadas.
- `GET /indicators/{ticker}`: indicadores técnicos por activo.
- `POST /signals/evaluate`: evaluación de señales técnicas.
- `POST /risk/var-cvar`: cálculo de VaR y CVaR.
- `POST /portfolio/markowitz`: optimización y frontera eficiente.
- `GET /capm/{ticker}`: métricas CAPM por activo.

### Machine Learning

- `POST /ml/predict`: predicción de señal de mercado.
- `POST /ml/risk-score`: score auxiliar de riesgo financiero.

### Modelos financieros avanzados

- `POST /volatility/ewma`: volatilidad y varianza EWMA.
- `POST /fixed-income/bond-metrics`: precio, duración, duración modificada y convexidad de un bono.
- `POST /fixed-income/nelson-siegel`: curva de tasas Nelson-Siegel.
- `POST /options/black-scholes`: precios teóricos call y put.
- `POST /options/greeks`: sensibilidades de opciones europeas.
- `POST /stress/portfolio`: choques por activo sobre retornos de portafolio.
- `POST /stress/combined-scenario`: escenario combinado adverso de precios, volatilidad y tasas.

Algunos endpoints dependen de datos externos o proveedores de mercado, como precios o variables macroeconómicas. Otros son completamente locales y determinísticos, como los modelos financieros analíticos, ML con artifacts versionados y pruebas unitarias.

## Frontend Streamlit

El frontend Streamlit no duplica fórmulas financieras. Consume el backend con `src/api/backend_client.py`, que resuelve la URL base desde `API_BASE_URL`, `BACKEND_API_BASE_URL`, `st.secrets` o el fallback local `http://127.0.0.1:8000`.

La interfaz muestra resultados, KPIs, tablas, gráficos e interpretación para los módulos del proyecto. El módulo de modelos financieros avanzados está implementado en `pages/10_modelos_financieros.py` y consume endpoints de EWMA, renta fija, Nelson-Siegel, Black-Scholes, Greeks y stress testing.

## Machine Learning

El endpoint `POST /ml/predict` usa un `RandomForestClassifier` entrenado con datos sintéticos reproducibles. Recibe variables técnicas simples (`close`, `sma`, `ema`, `rsi`) y devuelve una señal estimada: `Alcista`, `Bajista` o `Neutral`.

El endpoint `POST /ml/risk-score` usa `LogisticRegression` con `StandardScaler` para estimar un score auxiliar de riesgo a 5 días. El modelo usa features de retornos, volatilidad, RSI, MACD, Bandas de Bollinger, distancia a SMA y drawdown.

Ambos modelos se cargan como singleton/cache dentro del proceso FastAPI para evitar recargas innecesarias. Los artifacts `.joblib` están versionados en `models/`:

- `models/signal_classifier.joblib`
- `models/risk_classifier.joblib`

Estos modelos son académicos y usan datos sintéticos; sus salidas no constituyen recomendaciones financieras.

## Persistencia

El proyecto usa SQLAlchemy con SQLite. La base de datos local se configura con `DATABASE_URL` y por defecto apunta a `sqlite:///./data/risklab.db`.

El endpoint `GET /db/health` valida la conexión a la base de datos. Las predicciones de `POST /ml/predict` y los scores de `POST /ml/risk-score` se registran como logs técnicos para trazabilidad de entradas, salidas, versión de modelo y fecha de creación.

## Variables de entorno

Usa `.env.example` como referencia:

```env
FRED_API_KEY=tu_api_key
DEFAULT_START_DATE=2021-01-01

# Opcional: si se omite, el sistema usa la fecha actual.
# DEFAULT_END_DATE=2026-03-27

API_BASE_URL=http://127.0.0.1:8000
BACKEND_API_BASE_URL=http://127.0.0.1:8000
DATABASE_URL=sqlite:///./data/risklab.db
```

Variables principales:

- `FRED_API_KEY`: clave para datos macroeconómicos de FRED.
- `DEFAULT_START_DATE`: fecha inicial por defecto.
- `DEFAULT_END_DATE`: fecha final opcional para reproducir cortes históricos.
- `API_BASE_URL`: URL base del backend para el frontend.
- `BACKEND_API_BASE_URL`: variable compatible para apuntar al backend.
- `DATABASE_URL`: conexión SQLAlchemy/SQLite.

## Instalación local

Clonar el repositorio real:

```bash
git clone https://github.com/MariaAmaya12/Proyecto-Portafolio-Api-.git
cd Proyecto-Portafolio-Api-
```

Crear y activar entorno virtual en Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución local

Levantar backend FastAPI:

```bash
uvicorn backend.main:app --reload
```

Levantar frontend Streamlit en otra terminal:

```bash
streamlit run app.py
```

URLs locales:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:8501`
- Swagger: `http://127.0.0.1:8000/docs`
- Redoc: `http://127.0.0.1:8000/redoc`

## Docker

El proyecto incluye `Dockerfile`, `docker-compose.yml` y `.dockerignore`.

Ejecución con Docker Compose:

```bash
docker compose up --build
```

URLs esperadas:

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:8501`
- Swagger: `http://127.0.0.1:8000/docs`

Docker requiere tener Docker instalado y corriendo en el entorno local. En la validación final del proyecto, Docker no estaba disponible en el entorno de ejecución, por lo que la prueba real de contenedores quedó pendiente para una máquina con Docker.

## Pruebas y validación

Comandos de validación:

```bash
python -m compileall backend src pages tests
pytest
```

Resultado actual de la suite:

```text
46 passed
```

La validación final del proyecto está documentada en:

- `docs/fase7_validacion_final.md`

## CI

El workflow `.github/workflows/ci.yml` se ejecuta en `push` y `pull_request`. Sus pasos principales son:

- Checkout del repositorio.
- Configuración de Python 3.11.
- Instalación de dependencias desde `requirements.txt`.
- Ejecución de `python -m compileall backend src pages tests`.
- Ejecución de `pytest`.

## Documentación

Documentos técnicos existentes:

- `docs/contratos_api_frontend.md`
- `docs/teoria_modelos_financieros.md`
- `docs/fase5_machine_learning.md`
- `docs/fase5b_risk_score.md`
- `docs/fase7_validacion_final.md`

## Limitaciones

- El proyecto tiene fines académicos y no constituye recomendación financiera.
- Los modelos ML usan datos sintéticos y deben interpretarse como demostraciones metodológicas.
- Algunos datos dependen de proveedores externos y pueden fallar por disponibilidad, latencia, límites o conectividad.
- La validación Docker real debe ejecutarse en un entorno con Docker instalado y corriendo.
- La calidad de los resultados depende de los inputs, tickers, rango temporal, supuestos del modelo y disponibilidad de datos.

## Autores

**María Amaya y Esteban Díaz**

Proyecto académico de análisis financiero, gestión de riesgo, modelos cuantitativos e integración backend/frontend.
