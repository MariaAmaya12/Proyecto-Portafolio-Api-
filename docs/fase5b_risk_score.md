# Fase 5B — Risk Score Predictivo

## Responsable
Esteban — Backend FastAPI, Machine Learning, persistencia, pruebas y documentación técnica.

## Objetivo
Agregar una segunda capa de Machine Learning orientada a riesgo financiero, mediante un endpoint que estime la probabilidad de un evento adverso a 5 días.

---

## Diferencia frente a /ml/predict

| Endpoint | Propósito | Salida |
|---|---|---|
| `POST /ml/predict` | Clasifica una señal técnica de mercado | `Alcista`, `Bajista` o `Neutral` |
| `POST /ml/risk-score` | Estima probabilidad de pérdida relevante | `bajo`, `moderado` o `alto` |

`/ml/risk-score` tiene mayor valor financiero porque se enfoca en la probabilidad de que ocurra un evento adverso medible (caída acumulada > 2% en 5 días), mientras que `/ml/predict` describe una señal de tendencia sin cuantificar la magnitud del riesgo.

---

## Endpoint agregado

```
POST /ml/risk-score
```

**Request:**
```json
{
  "ret_1d": -0.005,
  "ret_5d": -0.018,
  "ret_20d": 0.012,
  "vol_5d": 0.014,
  "vol_20d": 0.011,
  "rsi": 42.0,
  "macd_hist": -0.003,
  "bb_position": 0.28,
  "close_over_sma20": 0.975,
  "drawdown_20d": -0.031
}
```

**Response:**
```json
{
  "risk_score": 0.73,
  "risk_level": "alto",
  "horizon_days": 5,
  "model_version": "risk-v1",
  "log_id": 1
}
```

| Campo | Descripción |
|---|---|
| `risk_score` | Probabilidad estimada de evento adverso (0.0 – 1.0) |
| `risk_level` | Nivel derivado: `bajo` (< 0.35), `moderado` (0.35 – 0.60), `alto` (> 0.60) |
| `horizon_days` | Horizonte de predicción en días (`5`) |
| `model_version` | Versión del modelo (`risk-v1`) |
| `log_id` | ID del registro en base de datos |

**Errores posibles:**
- `422` si algún campo tiene valor inválido.
- `503` si el modelo `.joblib` no está disponible o falla la persistencia.

---

## Features utilizadas

| Feature | Descripción |
|---|---|
| `ret_1d` | Retorno diario simple del activo |
| `ret_5d` | Retorno acumulado a 5 días |
| `ret_20d` | Retorno acumulado a 20 días |
| `vol_5d` | Volatilidad (desviación estándar) de retornos a 5 días |
| `vol_20d` | Volatilidad de retornos a 20 días |
| `rsi` | Relative Strength Index (0 – 100), sobrecompra/sobreventa |
| `macd_hist` | Histograma MACD normalizado por precio, indica momentum |
| `bb_position` | Posición del precio dentro de las Bandas de Bollinger (0 = límite inferior, 1 = superior) |
| `close_over_sma20` | Distancia relativa del precio frente a la SMA de 20 días |
| `drawdown_20d` | Caída máxima desde el máximo de 20 días (siempre ≤ 0) |

---

## Target financiero

El modelo se entrena para detectar eventos adversos definidos como:

```
forward_ret_5d[t] = prices[t+5] / prices[t] - 1
label[t] = 1  si  forward_ret_5d[t] < -0.02
label[t] = 0  en caso contrario
```

Es decir, la clase positiva representa una caída acumulada superior al 2% en los próximos 5 días de negociación.

---

## Dataset sintético

- Se simula una serie temporal de precios con **dos regímenes**:
  - **Normal** (85% del tiempo): media diaria `+0.03%`, volatilidad `1.0%`
  - **Estrés** (15% del tiempo): media diaria `-0.20%`, volatilidad `2.5%`
- No se usa internet ni `yfinance`.
- El generador usa `numpy.random.default_rng(42)`, garantizando reproducibilidad total.
- El objetivo es demostrar el pipeline ML aplicado a riesgo financiero, no producir un modelo con datos históricos reales.

---

## Prevención de data leakage

- Todas las features se calculan usando únicamente datos disponibles en tiempo `t` (ventanas `rolling`, `ewm`).
- El target usa información de `t+1` a `t+5`, empleada **solo como etiqueta**, nunca como feature.
- El split es **temporal**, no aleatorio:
  ```python
  cutoff = int(len(X) * 0.80)
  X_train, X_test = X[:cutoff], X[cutoff:]
  ```
- El `StandardScaler` se ajusta **exclusivamente** con `X_train` (`fit_transform`) y se aplica a `X_test` con `transform`, sin acceder a información futura.

---

## Modelo

- Algoritmo: `LogisticRegression` de scikit-learn.
- Parámetros: `class_weight="balanced"`, `C=0.1`, `max_iter=1000`, `random_state=42`.
- `class_weight="balanced"` compensa el desbalance de clases (≈ 22.6% positivos).
- El artifact guardado con `joblib` incluye:
  - `model` — clasificador entrenado
  - `scaler` — `StandardScaler` ajustado con `X_train`
  - `feature_names` — lista de 10 features en orden canónico
  - `model_version` — `"risk-v1"`
  - `horizon_days` — `5`
  - `threshold` — `-0.02`
  - `test_accuracy` — accuracy en el conjunto de prueba temporal
- Archivo generado: `models/risk_classifier.joblib`

---

## Predictor

`backend/ml/risk_predictor.py` expone la clase `RiskScorePredictor`:

- Carga el artifact con `joblib.load` en su constructor.
- Valida que existan todas las claves requeridas (`model`, `scaler`, `feature_names`, `model_version`, `horizon_days`).
- Verifica que `feature_names` coincida exactamente con `RISK_FEATURE_NAMES` para detectar desajustes entre el modelo guardado y el código.
- Valida que el modelo soporte `predict_proba`.
- La función `get_risk_predictor()` está decorada con `@lru_cache(maxsize=1)`, garantizando que el artifact se cargue **una sola vez** durante toda la vida del proceso FastAPI.
- El método `predict(payload)` devuelve:
  ```json
  {
    "risk_score": 0.73,
    "risk_level": "alto",
    "horizon_days": 5,
    "model_version": "risk-v1"
  }
  ```

---

## Persistencia

- Se agregó el modelo ORM `RiskScoreLog` (tabla `risk_score_logs`) en `backend/models.py`.
- Almacena los 10 features de entrada, `risk_score`, `risk_level`, `horizon_days`, `model_version` y `created_at`.
- La tabla se crea automáticamente al primer request mediante `create_database_tables()` (idempotente).
- El endpoint devuelve `log_id` con el ID del registro insertado.

---

## Pruebas

`tests/test_ml_risk_score.py` valida el contrato del endpoint sin depender de internet ni datos de mercado reales:

- **`test_risk_score_valid_request`** — verifica status `200`, presencia de los 5 campos (`risk_score`, `risk_level`, `horizon_days`, `model_version`, `log_id`), que `risk_score ∈ [0.0, 1.0]`, que `risk_level ∈ {bajo, moderado, alto}`, que `horizon_days == 5` y que `model_version` sea string no vacío.
- **`test_risk_score_invalid_request`** — verifica status `422` para payload inválido (`rsi="no-es-numero"`, campos faltantes).

---

## Validación

```
python -m compileall backend src pages tests   → sin errores
pytest                                          → 9/9 passed
```

Pruebas pasando al finalizar la fase:

| Archivo | Pruebas |
|---|---|
| `test_db_health.py` | 1 |
| `test_market_bundle.py` | 1 |
| `test_ml_predict.py` | 2 |
| `test_ml_risk_score.py` | 2 |
| `test_signal.py` | 3 |
| **Total** | **9** |

---

## Limitaciones

- El modelo usa datos sintéticos generados con reglas de dos regímenes; su accuracy refleja patrones artificiales, no series históricas reales.
- No debe interpretarse como recomendación financiera real.
- `risk_score` es un score auxiliar de riesgo; no reemplaza métricas cuantitativas como VaR o CVaR.
- Puede mejorarse en fases futuras con datos históricos reales (vía `yfinance`), validación walk-forward y un pipeline de re-entrenamiento periódico.

---

## Valor para el proyecto

Esta fase mejora el componente de IA del proyecto RiskLab porque conecta Machine Learning con riesgo financiero de forma directa: las features incluyen volatilidad, drawdown, RSI y retornos en múltiples horizontes, que son los mismos indicadores que alimentan los endpoints `POST /risk/var-cvar` y `GET /indicators/{ticker}`. Esto sienta las bases para integrar el score de riesgo ML con el panel de decisión del frontend, complementando las métricas cuantitativas existentes.

---

## Siguiente paso recomendado

Coordinar con María si se desea mostrar `POST /ml/risk-score` en Streamlit como una tarjeta de riesgo ML, usando el mismo payload que se puede derivar de los indicadores calculados por `GET /indicators/{ticker}`.
