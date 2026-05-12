# Fase 1 — Diagnóstico funcional del proyecto

## Proyecto

**RiskLab USTA — Aplicación de análisis de riesgo financiero**

## Responsable del diagnóstico

**María**

## Objetivo de la Fase 1

Validar el estado actual de la aplicación desarrollada en Streamlit, revisar su integración con el backend FastAPI, identificar problemas funcionales o visuales y dejar documentados los hallazgos necesarios para orientar las siguientes fases del proyecto.

Esta fase no tiene como objetivo implementar nuevos modelos financieros, sino dejar claro qué funciona, qué está incompleto y qué debe corregirse antes de avanzar con nuevas funcionalidades como SQLAlchemy, Machine Learning, renta fija, opciones y stress testing.

---

## 1. Estado general de la aplicación

Con base en las capturas revisadas, la aplicación se encuentra en un estado funcional y visualmente avanzado. La app carga correctamente en Streamlit, permite navegar entre módulos y presenta resultados financieros mediante KPIs, gráficos, tablas e interpretaciones.

La aplicación ya incluye módulos como:

- Contextualización del portafolio.
- Análisis técnico.
- Rendimientos.
- Modelos GARCH.
- CAPM y beta.
- VaR y CVaR.
- Optimización de portafolio con Markowitz.
- Señales técnicas.
- Macro y benchmark.
- Panel de decisión.

Sin embargo, todavía se identifica una arquitectura parcialmente híbrida: algunos módulos consumen el backend FastAPI, mientras que otros parecen mantener lógica local o descarga directa de datos. Esto debe corregirse en fases posteriores para que la aplicación funcione de manera más profesional y coherente.

---

## 2. Evaluación general por componente

| Componente | Estado actual | Observación |
|---|---|---|
| Streamlit | Funcional | La app carga, tiene navegación multipágina y diseño ordenado. |
| Backend FastAPI | Funcional | Swagger está disponible y expone endpoints financieros principales. |
| Integración frontend-backend | Parcial | Algunos módulos consumen backend y otros parecen usar flujo local. |
| Diseño visual | Bueno | Hay KPIs, tarjetas, gráficos y textos interpretativos. |
| Experiencia de usuario | Mejorable | La app aún se percibe en parte como dashboard por módulos. |
| Documentación técnica | Pendiente de fortalecer | Se requiere dejar trazabilidad de hallazgos y contratos API. |
| Cumplimiento nueva rúbrica | Parcial | Faltan SQLAlchemy, ML, renta fija, opciones, stress testing, Docker y CI. |

---

## 3. Hallazgos principales

### 3.1 La aplicación funciona, pero todavía comunica “dashboard”

En la página inicial aparece el enfoque de “Dashboard de Riesgo Financiero”. Dado que el profesor solicita una aplicación funcional y no únicamente un dashboard, se recomienda cambiar la narrativa visual y textual hacia una app guiada de análisis, riesgo y decisión financiera.

Recomendación:

```text
RiskLab USTA — Aplicación Integral de Riesgo Financiero
```

o

```text
RiskLab USTA — App de Análisis, Riesgo y Decisión Financiera
```

---

### 3.2 Hay integración parcial con backend

El proyecto ya cuenta con un cliente para consumir el backend y una estructura que permite obtener datos mediante FastAPI. Esto es positivo y debe mantenerse.

Sin embargo, se identificó que no todos los módulos parecen consumir el backend de forma homogénea. Algunas páginas muestran comportamiento diferente según el módulo, lo que puede indicar que todavía existen rutas de datos distintas.

Flujo deseado:

```text
Streamlit → cliente backend → FastAPI → servicios financieros → datos
```

Flujo que debe evitarse:

```text
Streamlit → yfinance o descarga directa
```

---

### 3.3 Problema detectado con el ticker 3382.T

Se observó que:

- Con horizonte de **1 año**, el ticker `3382.T` aparece como sin datos disponibles o es excluido del análisis conjunto.
- Con horizonte de **2 años**, la información sí carga correctamente.
- En algunos módulos individuales, como análisis técnico, el activo puede mostrar información aunque en el análisis conjunto sea excluido.

Esto sugiere que el problema no necesariamente es ausencia total de datos, sino una posible inconsistencia relacionada con:

- alineación de fechas entre activos,
- disponibilidad de datos por calendario bursátil,
- tratamiento de retornos faltantes,
- diferencias entre análisis individual y análisis conjunto,
- o uso de flujos de datos distintos entre páginas.

Este hallazgo debe ser tratado en la Fase 2.

---

### 3.4 El backend necesita devolver diagnósticos más claros

Actualmente la app informa que un ticker fue excluido, pero no siempre explica suficientemente el motivo.

Se recomienda que el backend devuelva una estructura más detallada como:

```json
{
  "missing_tickers": ["3382.T"],
  "calendar_diagnostics": {
    "3382.T": {
      "raw_rows": 240,
      "aligned_rows": 0,
      "reason": "insufficient aligned returns",
      "suggestion": "Use 2 years or analyze individually"
    }
  }
}
```

Esto permitiría que Streamlit muestre mensajes más claros al usuario.

---

## 4. Revisión funcional por módulo

| Módulo | Estado visual | Hallazgo principal | Acción futura |
|---|---|---|---|
| Inicio | Funcional | Presenta KPIs y resumen, pero comunica dashboard. | Cambiar narrativa hacia app. |
| Contextualización | Funcional | Buen soporte cualitativo del portafolio. | Conectar mejor con decisión final. |
| Análisis técnico | Funcional | Muestra KPIs, gráfico y señales. | Revisar si consume backend o descarga directa. |
| Rendimientos | Funcional | Buena explicación de distribución y normalidad. | Asegurar consumo desde backend. |
| GARCH | Funcional | Presenta volatilidad y persistencia. | Agregar advertencia si persistencia > 1 e incluir EWMA. |
| CAPM | Funcional | Buen uso de beta, alpha y R². | Reforzar interpretación cuando R² sea bajo. |
| VaR/CVaR | Funcional | Módulo fuerte de riesgo extremo. | Hacer visible el backtesting de Kupiec. |
| Markowitz | Funcional | Buena visualización de optimización. | Aclarar simulación vs optimización matemática. |
| Señales | Funcional | Interactivo y útil. | Revisar flujo backend y simplificar interpretación. |
| Macro y benchmark | Funcional | Aporta contexto macroeconómico. | Mostrar fuente de datos y fecha de actualización. |
| Panel de decisión | Funcional | Es el módulo que más se acerca a una app. | Explicar mejor cómo se genera la decisión. |

---

## 5. Fortalezas del estado actual

- La aplicación ya tiene una interfaz visual ordenada.
- Los módulos financieros clásicos están bastante avanzados.
- Existe backend FastAPI con Swagger.
- La app tiene panel de decisión, lo cual ayuda a que no sea solo descriptiva.
- Hay textos interpretativos en varios módulos.
- Se usan KPIs, gráficos y controles interactivos.
- El proyecto tiene potencial para convertirse en una app funcional completa.

---

## 6. Debilidades o riesgos

- La app todavía se percibe parcialmente como dashboard.
- Existen posibles diferencias de flujo entre módulos.
- El caso `3382.T` evidencia problemas de disponibilidad o alineación de datos.
- Algunas páginas podrían no estar consumiendo backend.
- Falta estado visible de conexión con backend.
- Falta configuración global del análisis.
- Faltan módulos exigidos en la nueva rúbrica: SQLAlchemy, ML, renta fija, opciones, stress testing, Docker y CI.
- El backend aún debe organizarse mejor con routers y tags en Swagger.

---

## 7. Prioridades para la siguiente fase

La Fase 2 debe enfocarse en:

1. Unificar la integración Streamlit-backend.
2. Revisar qué páginas todavía usan descarga directa.
3. Migrar páginas pendientes a `MarketDataClient` o `backend_client.py`.
4. Resolver o explicar técnicamente el caso `3382.T`.
5. Mejorar el mensaje de activos excluidos.
6. Mostrar estado de conexión al backend.
7. Documentar contratos de API para evitar conflictos entre María y Esteban.

---

## 8. Conclusión

La Fase 1 confirma que el proyecto tiene una base funcional sólida, pero todavía necesita ajustes de arquitectura y experiencia de usuario para cumplir mejor con la idea de una aplicación integral. La prioridad inmediata no debe ser agregar nuevos modelos, sino estabilizar el flujo frontend-backend y asegurar que todos los módulos consuman datos de forma consistente.
