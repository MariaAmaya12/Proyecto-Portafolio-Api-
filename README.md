
# Proyecto Integrador de Riesgo: Dashboard de Gestión de Portafolios con API

## Descripción general

Este proyecto desarrolla un dashboard interactivo para el análisis de un portafolio financiero, integrando métricas de riesgo, análisis técnico, modelos de volatilidad, CAPM, optimización de portafolios y contexto macroeconómico.

La solución está construida con una arquitectura separada en dos capas:

- **Frontend** en **Streamlit** para la visualización y exploración interactiva.
- **Backend** en **FastAPI** para exponer una API propia de datos y analítica.

El objetivo es que la interfaz visual no consuma directamente proveedores externos, sino que consulte primero una API interna controlada por el proyecto.

---

## Objetivo del proyecto

Construir una herramienta de análisis financiero que permita:

- visualizar la evolución histórica de activos del portafolio,
- calcular rendimientos y estadísticas descriptivas,
- estimar riesgo con VaR y CVaR,
- analizar volatilidad con modelos ARCH/GARCH,
- evaluar desempeño con CAPM y benchmark,
- optimizar portafolios con Markowitz,
- incorporar señales técnicas para apoyar decisiones de inversión,
- integrar variables macroeconómicas relevantes.

---

## Arquitectura del proyecto

La arquitectura actual sigue este flujo:

```text
Usuario
   ↓
Streamlit (frontend)
   ↓
FastAPI (backend propio)
   ↓
Servicios internos del proyecto
   ↓
Proveedores externos / cache macro
````

### Separación de responsabilidades

* **Streamlit**: interfaz visual y experiencia de usuario.
* **FastAPI**: capa API con endpoints tipados, validación, caching y manejo de errores.
* **src/services/**: lógica de acceso a datos de mercado y macroeconomía.
* **src/**: lógica analítica y financiera del proyecto.

Esto permite desacoplar el dashboard de APIs externas como Yahoo Finance, FRED o World Bank.

---

## Tecnologías utilizadas

* **Python**
* **Streamlit**
* **FastAPI**
* **Uvicorn**
* **Pydantic**
* **Pandas**
* **NumPy**
* **SciPy**
* **Plotly**
* **yfinance**
* **requests**
* **arch**
* **wbgapi**

---

## Estructura del proyecto

```text
riesgo_dashboard/
│
├── app.py
├── README.md
├── requirements.txt
├── .env
│
├── backend/
│   ├── main.py
│   └── cache.py
│
├── data/
│   └── macro_cache.json
│
├── pages/
│   ├── 0_contextualizacion.py
│   ├── 01_tecnico.py
│   ├── 02_rendimientos.py
│   ├── 03_garch.py
│   ├── 04_capm.py
│   ├── 05_var_cvar.py
│   ├── 06_markowitz.py
│   ├── 07_senales.py
│   ├── 08_macro_benchmark.py
│   └── 09_panel_decision.py
│
├── scripts/
│   └── update_macro_cache.py
│
├── src/
│   ├── api/
│   │   ├── backend_client.py
│   │   ├── market.py
│   │   └── macro.py
│   │
│   ├── services/
│   │   ├── market_service.py
│   │   └── macro_service.py
│   │
│   ├── benchmark.py
│   ├── capm.py
│   ├── config.py
│   ├── download.py
│   ├── garch_models.py
│   ├── indicators.py
│   ├── markowitz.py
│   ├── plots.py
│   ├── portfolio_optimization.py
│   ├── preprocess.py
│   ├── returns_analysis.py
│   ├── risk_metrics.py
│   └── signals.py
│
├── report/
│   └── informe_articulo.tex
│
├── .devcontainer/
│   └── devcontainer.json
│
└── .github/
    └── workflows/
        └── update_macro_cache.yml
```

---

## Funcionalidades principales

El dashboard incluye las siguientes secciones:

* **Contextualización** de activos y benchmark
* **Análisis técnico**
* **Rendimientos y propiedades empíricas**
* **Modelos ARCH/GARCH**
* **CAPM**
* **VaR / CVaR**
* **Optimización de Markowitz**
* **Señales automáticas**
* **Contexto macroeconómico y benchmark**
* **Panel final de decisión**

---

## API del backend

El backend expone **9 endpoints principales**:

| Método | Endpoint               | Descripción                                    |
| ------ | ---------------------- | ---------------------------------------------- |
| GET    | `/health`              | Verifica que la API esté disponible            |
| GET    | `/macro/snapshot`      | Devuelve el snapshot macroeconómico            |
| POST   | `/market/bundle`       | Devuelve precios, matriz de cierres y retornos |
| GET    | `/returns/{ticker}`    | Retornos, estadísticos descriptivos y pruebas  |
| GET    | `/indicators/{ticker}` | Indicadores técnicos por activo                |
| POST   | `/signals/evaluate`    | Evaluación de señales técnicas                 |
| POST   | `/risk/var-cvar`       | Cálculo de VaR y CVaR                          |
| POST   | `/portfolio/markowitz` | Optimización y frontera eficiente              |
| GET    | `/capm/{ticker}`       | Métricas CAPM por activo                       |

### Características técnicas del backend

* Validación de requests y responses con **Pydantic**
* Inyección de dependencias con **Depends()**
* Validación de pesos a nivel de schema y lógica de negocio
* Caching backend con TTL
* Manejo uniforme de errores HTTP
* Async/await aplicado de forma conservadora con `run_in_threadpool`

---

## Fuentes de datos

El proyecto usa fuentes externas de datos financieras y macroeconómicas, pero estas ya no son consumidas directamente por la interfaz visual.

Entre las fuentes utilizadas se encuentran:

* **Yahoo Finance**
* **FRED**
* **World Bank**
* **cache macro local/remoto**

La capa visual consume únicamente la **API interna del proyecto**.

---

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto con variables como estas:

```env
FRED_API_KEY=tu_api_key
DEFAULT_START_DATE=2021-01-01
# Opcional: si se omite, el sistema usa la fecha actual.
# Define DEFAULT_END_DATE solo para reproducir un corte histórico fijo.
# DEFAULT_END_DATE=2026-03-27
```

Si el frontend necesita apuntar al backend explícitamente, puede usarse también:

```env
API_BASE_URL=http://127.0.0.1:8000
```

> Si no se define `API_BASE_URL`, el cliente backend usa por defecto `http://127.0.0.1:8000` para desarrollo local.

Para Streamlit Community Cloud, configura la URL del backend de Render en **Settings > Secrets**:

```toml
API_BASE_URL="https://proyecto-portafolio-api-5ev7.onrender.com"
```

El frontend lee primero `st.secrets["API_BASE_URL"]`; si no existe, usa variables de entorno (`API_BASE_URL` o `BACKEND_API_BASE_URL` por compatibilidad).

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/MariaAmaya12/Proyecto-Portafolio-Api.git
cd Proyecto-Portafolio-Api
```

### 2. Crear entorno virtual

En Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar el proyecto

### Importante

Actualmente el sistema funciona con **dos procesos**:

1. **Backend FastAPI**
2. **Frontend Streamlit**

Debes levantar ambos para que el dashboard funcione correctamente.

---

### Opción recomendada: ejecutar en dos terminales

## Terminal 1: levantar backend

```bash
uvicorn backend.main:app --reload
```

Esto inicia la API en:

```text
http://127.0.0.1:8000
```

Y la documentación Swagger en:

```text
http://127.0.0.1:8000/docs
```

---

## Terminal 2: levantar frontend

```bash
streamlit run app.py
```

Esto inicia el dashboard en una URL similar a:

```text
http://localhost:8501
```

---

## Orden recomendado de ejecución

1. Activa el entorno virtual
2. Levanta **FastAPI**
3. Levanta **Streamlit**
4. Abre el dashboard en el navegador

---

## Cómo verificar que todo funciona

### Verificación del backend

Abre:

```text
http://127.0.0.1:8000/docs
```

Si ves Swagger con los 9 endpoints, el backend está funcionando.

### Verificación del frontend

Abre la URL que Streamlit muestre en consola.
Si el dashboard carga y muestra datos, entonces la integración está activa.

---

## Comportamiento esperado si el backend está apagado

Como el frontend ya está desacoplado de proveedores externos y consume el backend propio, si FastAPI no está corriendo pueden aparecer errores como:

* no fue posible descargar precios,
* fallo de conexión,
* datos no disponibles.

Esto es esperado, porque el dashboard ahora depende de la API interna.

---

## Ejemplo de pruebas rápidas

### Probar health

En Swagger o desde navegador:

```text
GET /health
```

### Probar snapshot macro

```text
GET /macro/snapshot
```

### Probar market bundle

```json
{
  "tickers": ["AAPL", "MSFT", "NVDA"],
  "start": "2024-01-01",
  "end": "2024-12-31"
}
```

### Probar VaR/CVaR con pesos válidos

```json
{
  "tickers": ["AAPL", "MSFT"],
  "start": "2024-01-01",
  "end": "2024-12-31",
  "weights": [0.6, 0.4],
  "alpha": 0.95,
  "n_sim": 10000
}
```

---

## Manejo de errores

La API devuelve errores con estructura uniforme:

```json
{
  "error": "Solicitud inválida.",
  "detail": [
    {
      "field": "weights",
      "message": "Value error, `weights` no puede estar vacío."
    }
  ]
}
```

Códigos utilizados:

* `400` parámetros inválidos
* `404` recurso no encontrado / sin datos
* `422` error de validación o datos insuficientes
* `502` fallo de proveedor externo
* `503` indisponibilidad temporal del servicio

---

## Caching

El proyecto usa varias capas de cache:

* **`st.cache_data`** en la capa Streamlit
* **TTL cache en backend** para respuestas y loaders
* **cache macro persistente** en `data/macro_cache.json`
* **workflow automático** para actualización de cache macro

Esto reduce descargas repetidas y mejora tiempo de respuesta.

---

## Archivo de actualización automática de cache macro

El proyecto incluye un workflow en:

```text
.github/workflows/update_macro_cache.yml
```

que actualiza `data/macro_cache.json` automáticamente.

También puedes actualizarlo manualmente con:

```bash
python scripts/update_macro_cache.py
```

---

## Recomendaciones de uso

* Levanta primero el backend y luego el frontend.
* Verifica que el archivo `.env` esté configurado.
* Si el dashboard no carga datos, revisa:

  * conexión a internet,
  * que FastAPI siga activo,
  * fechas válidas,
  * tickers válidos,
  * valor de `API_BASE_URL` en Streamlit Secrets o `.env`.

---

## Autora

**María Amaya**
Proyecto académico de análisis financiero, riesgo y optimización de portafolios.


