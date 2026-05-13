# Fase 4 — Docker y CI

## Responsable
Esteban — Infraestructura, despliegue local, automatización de pruebas y validación técnica.

## Objetivo
Agregar una configuración mínima de Docker y GitHub Actions CI para facilitar ejecución reproducible y validación automática del proyecto.

## Archivos creados
- `.dockerignore`
- `Dockerfile`
- `docker-compose.yml`
- `.github/workflows/ci.yml`

## Dockerfile
Se creó una imagen base con Python 3.11 slim. El proceso de build instala las dependencias de `requirements.txt`, copia el proyecto al contenedor y define como comando por defecto el arranque del servidor FastAPI con `uvicorn`.

## docker-compose.yml
Se definieron dos servicios:

- **backend** — FastAPI expuesto en el puerto `8000`.
- **frontend** — Streamlit expuesto en el puerto `8501`.

El servicio `frontend` recibe las siguientes variables de entorno para localizar el backend dentro de la red Docker:

```
API_BASE_URL=http://backend:8000
BACKEND_API_BASE_URL=http://backend:8000
```

Ambas variables se configuraron para mantener compatibilidad con el cliente backend del frontend (`src/api/backend_client.py`), que intenta resolver la URL del backend consultando primero `API_BASE_URL` y luego `BACKEND_API_BASE_URL`.

## .dockerignore
Se configuró para evitar que el build de Docker copie archivos innecesarios o sensibles:

- Entornos virtuales (`venv/`, `.venv/`)
- Cachés de Python (`__pycache__/`, `*.pyc`)
- Archivos de entorno (`.env`, `.env.*`)
- Bases de datos locales (`*.db`, `data/`)
- Datos y modelos generados
- Configuraciones de IDE (`.vscode/`, `.idea/`)

## GitHub Actions CI
El workflow `.github/workflows/ci.yml` valida automáticamente el proyecto en cada `push` y `pull_request`:

1. Checkout del repositorio.
2. Instalación de Python 3.11.
3. Upgrade de pip e instalación de `requirements.txt`.
4. Verificación de sintaxis Python con `python -m compileall backend src pages tests`.
5. Ejecución de la suite de pruebas con `pytest`.

## Validaciones locales
Ejecutadas en el entorno de desarrollo antes del commit:

```
python -m compileall backend src pages tests   → sin errores
pytest                                          → 5/5 passed
```

## Limitaciones
- `docker compose config` no pudo ejecutarse en el entorno de desarrollo porque Docker no estaba instalado. La validación de sintaxis del `docker-compose.yml` quedó pendiente de verificación visual.
- `docker compose up` debe validarse en una máquina local con Docker Desktop o un entorno equivalente con Docker instalado.
- Las pruebas del CI no dependen de internet ni de servicios externos.

## Siguiente paso recomendado
Revisar el resultado del workflow en GitHub Actions tras el push, confirmar que los 5 tests pasan en el runner de Ubuntu, y luego avanzar a Machine Learning o endpoints financieros faltantes según la rúbrica del proyecto.
