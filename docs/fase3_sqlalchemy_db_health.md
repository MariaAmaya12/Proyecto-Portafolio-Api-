# Fase 3 — SQLAlchemy, SQLite y /db/health

## Responsable

Esteban — Backend FastAPI, persistencia, configuración y pruebas.

## Objetivo

Agregar una capa mínima de persistencia usando SQLAlchemy y SQLite sin romper el backend existente.

## Cambios implementados

- Se agregó sqlalchemy a requirements.txt.
- Se agregó DATABASE_URL a .env.example.
- Se creó backend/database.py.
- Se creó backend/models.py.
- Se agregó GET /db/health.
- Se creó tests/test_db_health.py.

## Capa database

backend/database.py contiene:

- DATABASE_URL
- engine
- SessionLocal
- Base
- get_db
- check_database_connection

## Modelo ORM inicial

backend/models.py define SystemEvent con:

- id
- event_type
- message
- created_at

## Endpoint agregado

GET /db/health

Respuesta esperada:

```json
{
  "status": "ok",
  "database": "sqlite"
}
```

## Pruebas

tests/test_db_health.py valida:

- status_code 200
- status ok
- database sqlite

## Validación

- python -m compileall backend src pages tests
- pytest

## Limitaciones

Todavía no se crean tablas automáticamente ni se persisten eventos reales.
La capa creada prepara el proyecto para persistencia futura sin alterar módulos financieros.

## Siguiente paso recomendado

Avanzar a Docker y CI, o crear persistencia real para logs de predicción cuando se implemente Machine Learning.
