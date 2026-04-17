# Dashboard de Gestión de Portafolios y Teoría del Riesgo

Aplicación desarrollada en **Python** con una interfaz principal en **Streamlit** y una capa complementaria en **FastAPI** para exponer servicios de datos y facilitar pruebas manuales mediante Swagger. El proyecto integra datos históricos de mercado y variables macroeconómicas para evaluar rendimiento, riesgo, volatilidad, optimización y señales de decisión.

## Descripción

Este dashboard permite analizar un portafolio compuesto por activos internacionales mediante herramientas de análisis financiero y estadístico. La aplicación descarga datos desde APIs, construye métricas de riesgo y rendimiento, compara el portafolio frente a benchmarks y presenta los resultados en una interfaz interactiva.

La interfaz visual sigue funcionando en Streamlit. Además, la lógica de consulta de datos fue separada para permitir su reutilización desde un backend en FastAPI, lo que mejora la organización del proyecto y facilita futuras integraciones sin afectar el funcionamiento del dashboard.

## Objetivo

Desarrollar una herramienta interactiva para apoyar el análisis de portafolios desde la perspectiva de la teoría del riesgo, integrando:

- análisis técnico,
- rendimientos y estadística descriptiva,
- modelos ARCH/GARCH,
- CAPM,
- VaR y CVaR,
- optimización de Markowitz,
- señales automáticas,
- contexto macroeconómico y benchmark.

## Activos del portafolio

| Empresa | Ticker | Mercado |
|---|---|---|
| Seven & i Holdings | `3382.T` | Japón |
| Alimentation Couche-Tard | `ATD.TO` | Canadá |
| FEMSA | `FEMSAUBD.MX` | México |
| BP | `BP.L` | Reino Unido |
| Carrefour | `CA.PA` | Francia |

### Benchmark global

- `ACWI`

### Benchmarks locales usados en CAPM

- Seven & i Holdings → `^N225`
- Alimentation Couche-Tard → `^GSPTSE`
- FEMSA → `^MXX`
- BP → `^FTSE`
- Carrefour → `^FCHI`

## Tecnologías utilizadas

- Python
- Streamlit
- FastAPI
- Uvicorn
- pandas
- numpy
- scipy
- plotly
- yfinance
- arch
- requests
- python-dotenv
- wbgapi

## Arquitectura

La arquitectura actual separa interfaz, wrappers y lógica reutilizable:

### 1. Streamlit

La aplicación principal del dashboard se ejecuta desde `app.py` y organiza sus vistas en la carpeta `pages/`.

### 2. `src/api/`

Contiene wrappers compatibles con Streamlit. Su función es conservar una interfaz estable para el dashboard y aplicar cache cuando corresponde.

Ejemplos:

- `src.api.macro.macro_snapshot()`
- `src.api.market.get_prices()`
- `src.api.market.get_multiple_prices()`

### 3. `src/services/`

Contiene la lógica de negocio desacoplada de Streamlit.

- `macro_service.py`: snapshot macro, cache remoto y fallbacks.
- `market_service.py`: descarga de precios, limpieza, validación y construcción de matrices.

### 4. `backend/main.py`

Expone el backend FastAPI del proyecto y reutiliza la lógica de `src/services/`.

Swagger se usa únicamente como herramienta de documentación y de prueba manual de los endpoints.

## Estructura del proyecto

```text
riesgo_dashboard/
├── app.py
├── backend/
│   └── main.py
├── requirements.txt
├── README.md
├── pages/
│   ├── 01_tecnico.py
│   ├── 02_rendimientos.py
│   ├── 03_garch.py
│   ├── 04_capm.py
│   ├── 05_var_cvar.py
│   ├── 06_markowitz.py
│   ├── 07_senales.py
│   └── 08_macro_benchmark.py
├── src/
│   ├── api/
│   │   ├── macro.py
│   │   └── market.py
│   ├── services/
│   │   ├── macro_service.py
│   │   └── market_service.py
│   ├── benchmark.py
│   ├── capm.py
│   ├── config.py
│   ├── download.py
│   ├── garch_models.py
│   ├── indicators.py
│   ├── markowitz.py
│   ├── plots.py
│   ├── preprocess.py
│   ├── returns_analysis.py
│   ├── risk_metrics.py
│   └── signals.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── macro_cache.json
├── scripts/
│   └── update_macro_cache.py
└── report/
    └── informe_articulo.tex
```

## Módulos del dashboard

### 1. Análisis técnico

Visualiza precios, medias móviles, RSI, MACD, Bandas de Bollinger y oscilador estocástico.

### 2. Rendimientos

Calcula rendimientos simples y logarítmicos, además de estadísticas descriptivas y comportamiento acumulado.

### 3. Modelos GARCH

Permite analizar la volatilidad condicional mediante modelos ARCH/GARCH.

### 4. CAPM

Estima beta, alpha y retorno esperado del activo respecto a su benchmark.

### 5. VaR y CVaR

Calcula métricas de riesgo extremo mediante distintos enfoques.

### 6. Markowitz

Simula portafolios, construye la frontera eficiente y obtiene portafolios de mínima varianza y máximo Sharpe.

### 7. Señales y alertas

Genera recomendaciones automáticas de compra, venta o mantener a partir de indicadores técnicos.

### 8. Macro y benchmark

Integra variables macroeconómicas y compara el portafolio frente al benchmark global.

## Backend FastAPI

El proyecto incluye un backend complementario para exponer datos y facilitar validaciones manuales desde Swagger.

### Endpoints actuales

#### `GET /health`

Verifica que el backend esté activo.

Respuesta esperada:

```json
{
  "status": "ok"
}
```

#### `GET /macro/snapshot`

Devuelve un snapshot macroeconómico consolidado con campos como:

- `risk_free_rate_pct`
- `inflation_yoy`
- `cop_per_usd`
- `usdcop_market`
- `source`
- `last_updated`

#### `POST /market/bundle`

Devuelve un bundle de mercado con:

- `ohlcv` por ticker,
- matriz `close`,
- matriz `returns`.

Ejemplo de request válido:

```json
{
  "tickers": ["AAPL", "MSFT", "^GSPC"],
  "start": "2024-01-01",
  "end": "2024-12-31"
}
```

El endpoint valida:

- que la lista de tickers no esté vacía,
- que cada ticker sea válido,
- que las fechas tengan formato correcto,
- que `end` no sea anterior a `start`,
- que existan datos para el rango solicitado.

## Verificación manual en Swagger

Durante la validación local del backend se comprobaron manualmente los siguientes casos desde Swagger.

### 1. Respuesta exitosa de `POST /market/bundle`

Ejemplo de referencia:

```json
{
  "ohlcv": {
    "AAPL": [
      {
        "Date": "2024-01-02T00:00:00",
        "Open": 187.15,
        "High": 188.44,
        "Low": 183.89,
        "Close": 185.64,
        "Adj Close": 184.94,
        "Volume": 82488700
      }
    ]
  },
  "close": [
    {
      "Date": "2024-01-02T00:00:00",
      "AAPL": 184.94,
      "MSFT": 368.51,
      "^GSPC": 4742.83
    }
  ],
  "returns": [
    {
      "Date": "2024-01-03T00:00:00",
      "AAPL": -0.0075,
      "MSFT": -0.0014,
      "^GSPC": -0.0080
    }
  ]
}
```

### 2. Error 422 por request inválido

Ejemplo de referencia:

```json
{
  "error": "Solicitud inválida.",
  "detail": [
    {
      "field": "tickers[1]",
      "message": "String should have at least 1 character"
    }
  ]
}
```

### 3. Error 404 por ticker inexistente o sin datos

Ejemplo de referencia:

```json
{
  "error": "No se encontraron datos para uno o más tickers.",
  "detail": [
    {
      "field": "tickers[0]",
      "message": "No se pudo descargar datos para 'STRING' o no hubo precios en el rango solicitado."
    }
  ]
}
```

## Flujo general del sistema

```text
config -> services/api -> download/preprocess -> análisis/métricas -> visualización -> páginas del dashboard
```

## Fuentes de datos

### Yahoo Finance

Se utiliza para descargar precios históricos de los activos y benchmarks.

### FRED

Se utiliza para obtener variables macroeconómicas, por ejemplo:

- `DGS3MO` → tasa libre de riesgo
- `CPIAUCSL` → inflación
- `COLCCUSMA02STM` → tipo de cambio COP/USD

### World Bank

Se utiliza como fuente de respaldo en algunos datos macroeconómicos cuando FRED no responde o no entrega la serie esperada.

## Requisitos

- Python 3.10 o superior
- Conexión a internet
- API key de FRED para habilitar completamente el módulo macroeconómico

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/MariaAmaya12/Proyecto-Portafolio-Api-.git
cd Proyecto-Portafolio-Api-
```

### 2. Crear y activar el entorno virtual

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecución local

### Streamlit

```bash
python -m streamlit run app.py
```

Luego abre:

```text
http://localhost:8501
```

### FastAPI

```bash
python -m uvicorn backend.main:app --reload
```

Luego abre:

```text
http://127.0.0.1:8000
```

Documentación interactiva:

```text
http://127.0.0.1:8000/docs
```


## Autoría

Proyecto académico orientado al análisis de portafolios, teoría del riesgo y visualización financiera.
