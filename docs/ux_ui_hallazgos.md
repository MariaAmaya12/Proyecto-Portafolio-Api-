# Hallazgos UX/UI — Fase 1

## Objetivo

Evaluar la interfaz visual, experiencia de usuario y percepción general de la aplicación con base en las capturas revisadas. El objetivo es identificar ajustes que permitan que el proyecto se perciba como una aplicación funcional y no únicamente como un dashboard financiero.

---

## 1. Evaluación general de diseño

La aplicación tiene una interfaz limpia, ordenada y visualmente profesional. Se observan tarjetas de KPIs, gráficos financieros, controles laterales, tablas, paneles interpretativos y secciones explicativas.

La estética actual es adecuada para un proyecto académico avanzado. Sin embargo, todavía conserva características de dashboard tradicional, especialmente por la organización en módulos, el uso del título “Dashboard” y la falta de un flujo guiado de usuario.

---

## 2. Puntos fuertes visuales

- Uso claro de tarjetas KPI.
- Sidebar con navegación entre módulos.
- Gráficos financieros bien distribuidos.
- Textos interpretativos en varios módulos.
- Panel de decisión como cierre del análisis.
- Controles interactivos para horizonte, activo, simulaciones y umbrales.
- Visualización clara de métricas como beta, VaR, CVaR, Sharpe, RSI y volatilidad.
- La app permite exportar datos en algunos módulos, lo cual aporta sensación funcional.

---

## 3. Problemas visuales o de experiencia

### 3.1 El título comunica dashboard

Actualmente la aplicación se presenta como un dashboard de riesgo financiero. Dado que el profesor solicita una aplicación funcional, se recomienda cambiar el lenguaje.

Texto sugerido:

```text
RiskLab USTA — Aplicación Integral de Riesgo Financiero
```

o

```text
RiskLab USTA — App de Análisis, Riesgo y Decisión Financiera
```

Este cambio es simple, pero modifica la percepción general del proyecto.

---

### 3.2 La navegación parece organizada como entrega académica

Los nombres con “Mód.1”, “Mód.2”, etc. son claros para identificar entregables, pero visualmente refuerzan la idea de dashboard o informe por capítulos.

Se recomienda usar nombres orientados al usuario:

| Nombre actual | Nombre sugerido |
|---|---|
| Mód.1 Análisis técnico | Señales técnicas |
| Mód.2 Rendimientos | Rendimientos y distribución |
| Mód.3 Modelos GARCH | Volatilidad condicional |
| Mód.4 CAPM y Beta | Riesgo sistemático |
| Mód.5 VaR/CVaR | Riesgo extremo |
| Mód.6 Optimización Markowitz | Optimización de portafolio |
| Mód.7 Señales | Alertas operativas |
| Mód.8 Macro y Benchmark | Contexto macro y benchmark |
| Mód.9 Panel de decisión | Decisión integrada |

---

### 3.3 Falta configuración global del análisis

Cada módulo parece manejar sus propios filtros o controles. Esto funciona, pero hace que la aplicación se sienta como un conjunto de páginas independientes.

Para que se perciba más como app, debería existir una configuración global:

```text
Activos seleccionados
Horizonte de análisis
Benchmark
Nivel de confianza
Número de simulaciones
Portafolio seleccionado
```

Técnicamente, esto puede manejarse con `st.session_state`.

---

### 3.4 Falta estado visible del backend

La app debería mostrar si el backend está conectado.

Ejemplo visual:

```text
Backend conectado
URL backend: https://...
Última actualización: ...
```

Esto mejora la confianza del usuario y facilita la sustentación.

---

### 3.5 Mensajes de error mejorables

Cuando un ticker se excluye, el mensaje debe ser más explicativo.

Mensaje actual esperado:

```text
Sin datos para estos tickers en el rango seleccionado: 3382.T
```

Mensaje sugerido:

```text
3382.T fue excluido del análisis conjunto porque no tiene suficientes retornos alineados con los demás activos en el horizonte seleccionado. Puedes ampliar el horizonte a 2 años o analizarlo individualmente.
```

---

## 4. Diferencia entre app y dashboard

### Elementos que actualmente parecen dashboard

- Navegación por módulos.
- KPIs y gráficos independientes.
- Filtros por página.
- Estructura de análisis separada por temas.
- Título con la palabra “Dashboard”.

### Elementos que sí se acercan a una app

- Botón de actualización de datos.
- Controles interactivos.
- Parámetros configurables.
- Exportación de datos.
- Panel de decisión.
- Integración parcial con backend.
- Cálculos financieros reales y no solo visualización.

---

## 5. Recomendación para que se perciba como aplicación funcional

La app debería seguir un flujo más guiado:

```text
1. Configurar análisis
2. Validar datos disponibles
3. Ejecutar análisis financiero
4. Comparar riesgos
5. Optimizar portafolio
6. Evaluar escenarios
7. Generar decisión integrada
```

Actualmente la app permite navegar entre módulos, pero no siempre guía al usuario sobre cuál es el siguiente paso.

---

## 6. Mejoras sugeridas por sección

### Inicio

- Cambiar “Dashboard” por “Aplicación”.
- Mostrar estado de conexión al backend.
- Mostrar activos válidos y activos excluidos.
- Mostrar fecha de última actualización.
- Renombrar “Actualizar datos” como “Ejecutar análisis”.

### Contextualización

- Mantener la explicación de activos.
- Agregar una columna de rol dentro del portafolio:
  - defensivo,
  - cíclico,
  - energético,
  - diversificador,
  - consumo básico.

### Análisis técnico

- Mantener KPIs y gráficos.
- Asegurar que use backend.
- Mostrar interpretación más directa de señal actual.

### Rendimientos

- Mantener histograma, boxplot y Q-Q plot.
- Agregar interpretación automática cuando se rechace normalidad.
- Relacionar el resultado con VaR y CVaR.

### GARCH

- Agregar advertencia si la persistencia es mayor que 1.
- Incluir comparación con EWMA en fases posteriores.
- Explicar mejor limitaciones del pronóstico.

### CAPM

- Mantener beta, alpha y R².
- Si R² es bajo, mostrar una advertencia:
  “El benchmark explica poco la variabilidad del activo en este periodo.”

### VaR/CVaR

- Hacer visible el backtesting de Kupiec.
- Diferenciar claramente VaR paramétrico, histórico y Monte Carlo.
- Mostrar conclusión ejecutiva del riesgo.

### Markowitz

- Aclarar si el resultado viene de simulación o de optimización matemática.
- Mostrar pesos finales de forma más destacada.
- Agregar explicación breve de frontera eficiente.

### Señales

- Mostrar primero la señal actual.
- Luego mostrar detalle histórico.
- Evitar saturar el gráfico con demasiadas marcas.

### Macro y benchmark

- Mostrar fuente de datos.
- Mostrar fecha de actualización.
- Explicar cómo la tasa libre de riesgo alimenta CAPM y Markowitz.

### Panel de decisión

- Fortalecer la lógica de decisión.
- Mostrar razones por componente:
  - riesgo,
  - técnica,
  - benchmark,
  - optimización,
  - macro.
- Agregar conclusión descargable o copiable.

---

## 7. Componentes UI recomendados

Para mejorar mantenibilidad, se recomienda crear una carpeta:

```text
src/ui/
```

Con archivos como:

```text
src/ui/cards.py
src/ui/messages.py
src/ui/sidebar.py
src/ui/layout.py
```

Funciones sugeridas:

```python
render_kpi_card(...)
render_backend_status(...)
render_excluded_ticker_warning(...)
render_interpretation_box(...)
render_section_header(...)
```

Esto evitaría repetir código visual en varias páginas.

---

## 8. Priorización UX/UI

### Crítico

1. Cambiar “Dashboard” por “Aplicación”.
2. Mostrar estado del backend.
3. Explicar mejor activos excluidos.
4. Unificar el flujo de datos con backend.
5. Crear una lógica de app guiada.

### Importante

1. Renombrar navegación.
2. Crear configuración global con `st.session_state`.
3. Mejorar panel de decisión.
4. Mostrar fuente de datos.
5. Agregar advertencias teóricas en GARCH, CAPM y VaR.

### Opcional

1. Agregar íconos.
2. Agregar barra de progreso.
3. Agregar tema visual más personalizado.
4. Agregar exportación de reporte.
5. Agregar modo oscuro.

---

## 9. Conclusión UX/UI

La aplicación tiene una base visual sólida y funcional. Ya supera un dashboard básico porque permite interacción, análisis y decisión. Sin embargo, todavía conserva una estructura modular típica de dashboard. Para cumplir mejor con la solicitud del profesor, se recomienda transformar la narrativa, el flujo y la navegación para que el usuario perciba una aplicación integral de análisis de riesgo financiero.
