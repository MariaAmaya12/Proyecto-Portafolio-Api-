# Fase 5 — Machine Learning

## Responsable
Esteban — Backend FastAPI, pipeline ML, persistencia y pruebas.

## Objetivo
Agregar un pipeline mínimo de Machine Learning clásico, reproducible y testeable, servido mediante un endpoint REST.

## Dependencias agregadas
- `scikit-learn>=1.4.0`
- `joblib>=1.3.0`

## Estructura creada
- `backend/ml/__init__.py` — marca el paquete `backend.ml`
- `backend/ml/features.py` — validación y extracción de features, generación de dataset sintético
- `backend/ml/train.py` — entrenamiento reproducible con `RandomForestClassifier` y guardado con `joblib`
- `backend/ml/predictor.py` — carga única del modelo mediante `lru_cache`
- `models/.gitkeep` — versiona el directorio `models/` en Git
- `models/signal_classifier.joblib` — modelo oficial pre-entrenado (versionado en repo)
- `tests/test_ml_predict.py` — pruebas del endpoint sin internet

## Dataset sintético
El entrenamiento utiliza datos sintéticos generados localmente con `numpy.random.default_rng(42)`, sin depender de internet ni de `yfinance`. Se producen 1000 muestras con features `close`, `sma`, `ema` y `rsi`, etiquetadas automáticamente según reglas de señal (Alcista / Bajista / Neutral). El uso de `random_state=42` garantiza reproducibilidad total: reentrenar el modelo siempre produce el mismo archivo `.joblib`.

## Modelo entrenado
Se entrenó un `RandomForestClassifier` con `n_estimators=100` y `class_weight="balanced"`, usando `train_test_split` con estratificación para respetar la distribución de clases. El modelo alcanzó una **accuracy del 91.5%** en el test set y fue guardado con `joblib` en:

```
models/signal_classifier.joblib
```

El artifact contiene el modelo, la versión (`v1`), los nombres de features y la accuracy de entrenamiento.

## Predictor
`backend/ml/predictor.py` expone la clase `SignalPredictor`, que carga el archivo `.joblib` en su constructor y valida su existencia con un `FileNotFoundError` descriptivo si no se encuentra. La función `get_predictor()` está decorada con `@lru_cache(maxsize=1)` para garantizar que el modelo se cargue **una sola vez** durante toda la vida del proceso FastAPI, evitando recargas innecesarias en cada predicción.

## Endpoint agregado

```
POST /ml/predict
```

**Request esperado:**
```json
{
  "close": 110.0,
  "sma": 100.0,
  "ema": 105.0,
  "rsi": 55.0
}
```

**Response esperado:**
```json
{
  "prediction": "Alcista",
  "probability": 0.87,
  "model_version": "v1",
  "log_id": 1
}
```

**Manejo de errores:**
- `422` si los features son inválidos (p. ej. `rsi > 100`, valor no numérico).
- `503` si el modelo `.joblib` no existe o falla la predicción.
- `503` si falla la persistencia en base de datos.

## Persistencia
Cada predicción se registra en SQLite mediante el modelo ORM `PredictionLog` (tabla `prediction_logs`), que almacena los features de entrada (`close`, `sma`, `ema`, `rsi`), la clase predicha, la probabilidad, la versión del modelo y la marca de tiempo. La tabla se crea automáticamente al primer request mediante `create_database_tables()`, que utiliza `Base.metadata.create_all()` de SQLAlchemy de forma idempotente.

## Pruebas
`tests/test_ml_predict.py` valida el contrato del endpoint sin depender de internet ni de datos de mercado reales:

- **`test_ml_predict_valid_request`** — verifica status `200`, presencia de todos los campos (`prediction`, `probability`, `model_version`, `log_id`), que `prediction ∈ {Alcista, Bajista, Neutral}`, que `probability ∈ [0.0, 1.0]` y que `model_version` sea string no vacío.
- **`test_ml_predict_invalid_request`** — verifica status `422` para payload inválido (`close="no-es-numero"`).

## Validación

```
python -m compileall backend src pages tests   → sin errores
pytest                                          → 7/7 passed
```

Pruebas pasando al finalizar la fase:

| Archivo | Pruebas |
|---|---|
| `test_db_health.py` | 1 |
| `test_market_bundle.py` | 1 |
| `test_ml_predict.py` | 2 |
| `test_signal.py` | 3 |
| **Total** | **7** |

## Limitaciones
- El modelo usa datos sintéticos generados con reglas simples; su accuracy (91.5%) refleja patrones artificiales, no históricos de mercado.
- No debe interpretarse como recomendación financiera real.
- El objetivo de la fase es demostrar la integración completa de ML + API REST + persistencia en base de datos dentro del proyecto RiskLab USTA.
- Puede mejorarse en fases futuras con datos históricos reales (vía `yfinance`), validación cruzada más robusta y un pipeline de re-entrenamiento automatizado.

## Siguiente paso recomendado
Coordinar con María para consumir `POST /ml/predict` desde Streamlit, o continuar con los endpoints financieros faltantes de la rúbrica del proyecto.
