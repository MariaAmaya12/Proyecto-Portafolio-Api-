# Teoria de modelos financieros - RiskLab USTA

## 1. Introduccion

RiskLab USTA integra modelos cuantitativos para analizar riesgo de mercado, rendimiento,
volatilidad, renta fija, opciones y escenarios adversos. El objetivo de estos modelos es
apoyar la interpretacion financiera dentro de un entorno academico, reproducible y
testeable.

Las funciones implementadas en `src/` son puras: reciben datos, validan entradas y
devuelven resultados numericos o estructuras reutilizables. Esto permite que el backend
FastAPI y el frontend Streamlit las consuman sin duplicar logica financiera.

## 2. Volatilidad historica, EWMA y GARCH

La volatilidad historica mide la dispersion de los rendimientos observados durante una
ventana de tiempo. Es facil de interpretar, pero asigna el mismo peso a observaciones
recientes y antiguas dentro de la muestra.

EWMA, o promedio movil exponencialmente ponderado, asigna mayor peso a los datos mas
recientes. Su parametro lambda controla la persistencia: valores altos, como 0.94,
suavizan mas la serie y conservan memoria de observaciones pasadas; valores menores
reaccionan con mayor rapidez a cambios recientes. Por eso EWMA es util cuando se busca
capturar aumentos o caidas recientes de volatilidad sin estimar un modelo mas complejo.

GARCH modela la volatilidad condicional como un proceso dinamico donde la varianza actual
depende de choques pasados y de varianzas previas. A diferencia de EWMA, GARCH estima
parametros y permite representar agrupamiento de volatilidad. En RiskLab USTA, EWMA y
GARCH se complementan: EWMA ofrece una medida simple y transparente, mientras que GARCH
aporta una lectura econometrica mas estructurada.

## 3. Renta fija

El precio de un bono corresponde al valor presente de sus flujos futuros: cupones y
valor nominal al vencimiento. Si la tasa de mercado sube, el valor presente de esos flujos
disminuye; si la tasa baja, el precio del bono aumenta.

La curva de rendimiento relaciona tasas de interes con diferentes vencimientos. Puede
usarse para interpretar expectativas de mercado, prima por plazo y sensibilidad de los
instrumentos de renta fija.

Nelson-Siegel es una especificacion paramatrica para representar la curva de rendimiento.
Combina tres componentes: nivel de largo plazo, pendiente de corto plazo y curvatura de
mediano plazo. Su ventaja es que permite construir curvas suaves con pocos parametros,
incluso cuando se trabaja con escenarios sinteticos o academicos.

La duracion de Macaulay mide el plazo promedio ponderado de los flujos de un bono. La
duracion modificada transforma esa medida en sensibilidad aproximada del precio ante
cambios en la tasa de mercado. La convexidad corrige la aproximacion lineal de la
duracion, especialmente cuando los cambios de tasa son grandes.

## 4. Opciones

Una opcion call europea otorga el derecho, pero no la obligacion, de comprar un activo a
un precio de ejercicio en la fecha de vencimiento. Una opcion put europea otorga el
derecho de vender el activo bajo las mismas condiciones.

El modelo Black-Scholes permite valorar opciones europeas bajo supuestos especificos:
mercados sin fricciones, tasa libre de riesgo constante, volatilidad constante, ausencia
de arbitraje y rendimientos lognormales del activo subyacente. Aunque estos supuestos no
siempre se cumplen en mercados reales, el modelo es una referencia central para docencia,
comparacion y analisis de sensibilidad.

## 5. Greeks

Las Greeks miden sensibilidades del precio de una opcion:

- Delta: cambio esperado del precio de la opcion ante un cambio pequeno en el precio del
  subyacente.
- Gamma: cambio del Delta ante variaciones del subyacente; mide curvatura.
- Vega: sensibilidad del precio de la opcion ante cambios en la volatilidad.
- Theta: sensibilidad frente al paso del tiempo.
- Rho: sensibilidad ante cambios en la tasa libre de riesgo.

Estas medidas ayudan a interpretar exposiciones, no garantizan resultados futuros. Su
lectura depende de los supuestos del modelo y de la calidad de los datos de entrada.

## 6. Stress testing

El stress testing evalua el comportamiento de precios, retornos o portafolios bajo
escenarios adversos definidos por el analista. En RiskLab USTA se consideran:

- Shock de precio: aplica una caida o aumento porcentual sobre precios.
- Shock de volatilidad: multiplica los rendimientos para simular mayor incertidumbre.
- Shock de tasa: estima el impacto en un bono usando duracion modificada y, cuando esta
  disponible, convexidad.
- Escenario combinado adverso: integra varios shocks para observar efectos simultaneos en
  precios, retornos, bonos y portafolios.

VaR y stress testing responden preguntas distintas. El VaR estima una perdida potencial
con un nivel de confianza usando datos historicos, supuestos parametricos o simulaciones.
El stress testing no depende necesariamente de probabilidades historicas; pregunta que
ocurriria bajo escenarios especificos, aunque sean extremos o hipoteticos.

## 7. Machine Learning como complemento

El endpoint `/ml/risk-score` es una herramienta auxiliar para clasificar o resumir senales
de riesgo a partir de variables financieras. No reemplaza VaR, GARCH, CAPM, Markowitz,
Black-Scholes ni el criterio financiero del analista.

El aprendizaje automatico puede aportar patrones y alertas, pero sus salidas deben leerse
como apoyo interpretativo. En un sistema academico, el ML es mas valioso cuando se
contrasta con medidas financieras transparentes y con explicaciones del contexto de
mercado.

## 8. Limitaciones financieras del sistema

RiskLab USTA tiene fines academicos y no constituye recomendacion financiera, legal ni de
inversion. Los resultados dependen de los datos disponibles, de su calidad y de las
ventanas de analisis seleccionadas.

Los modelos usan supuestos simplificadores. EWMA depende de lambda, GARCH depende de la
especificacion estimada, Black-Scholes supone volatilidad constante y Nelson-Siegel puede
representar curvas suaves pero no garantiza ajuste perfecto a mercados reales.

Cuando se usen datos sinteticos, especialmente en componentes de Machine Learning, los
resultados deben interpretarse como demostraciones metodologicas y no como evidencia
empirica concluyente. La validacion financiera requiere datos reales, pruebas de robustez
y contraste con criterio experto.
