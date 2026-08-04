# Datasets · Introducción a Machine Learning (edición 2026)

Cada dataset declara su **origen** (de uso público con fuente y licencia, o dato sintético generado
para el curso) y en qué clase se usa. Las columnas están **en español y en `snake_case`**; cuando el
dataset original viene en inglés, acá queda documentada la equivalencia.

| Dataset | Filas × cols | Origen | Usado en |
|:--|:--:|:--|:--|
| `morosidad_cartera.csv` | 600 × 8 | Público — UCI ML Repository, *Default of Credit Card Clients* (CC BY 4.0) | Clase 1 |
| `absentismo_laboral.csv` | 740 × 22 | Público — UCI ML Repository, *Absenteeism at Work* (CC BY 4.0) | Clase 2 |
| `cobranza_servicios.csv` | 612 × 17 | **Sintético**, generado para el curso | Clase 3 |

---

## `morosidad_cartera.csv`

Cartera de clientes de **tarjeta de crédito**: comportamiento de pago de un mes y si el cliente
**incumplió el pago del mes siguiente**. En el material de clase se habla de crédito de consumo,
que es el encuadre de gestión; el registro original es de tarjetas y por eso `monto_facturado_mes`
es un estado de cuenta. Sirve para reconocer, sobre un mismo caso, los distintos
encuadres de un problema (clasificación, regresión, segmentación y preguntas que no son predictivas).

### Origen y licencia

- **Fuente:** UCI Machine Learning Repository, *Default of Credit Card Clients* (dataset 350) —
  <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0) —
  <https://creativecommons.org/licenses/by/4.0/>
- **Obra derivada:** esta es una versión **modificada** del original. Se tomó una submuestra de
  600 filas y 8 de las 25 columnas, con las columnas traducidas al español y algunas
  recodificadas. El detalle está en «Preparación aplicada», más abajo.
- **Cita original:** Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for
  the predictive accuracy of probability of default of credit card clients. *Expert Systems with
  Applications*, 36(2), 2473–2480.
- **Datos reales y anonimizados**: clientes de una entidad financiera de Taiwán, año 2005. No
  contiene identificadores personales; `id_cliente` se reasignó en esta versión.

### Preparación aplicada (reproducible)

El script que la genera está en `_prep_morosidad_cartera.py`, junto a este README.

Sobre el `.xls` original (30.000 × 25), leído con `header=1`:

1. Se conservaron 8 columnas de las 25 y se renombraron al español (tabla de equivalencias abajo).
2. `EDUCATION` se recodificó de números a etiquetas: `1 → posgrado`, `2 → universitaria`,
   `3 → media`, y `0, 4, 5, 6 → otra`.
3. `PAY_0` (estado de pago del mes, con valores negativos que indican pago al día o sin consumo) se
   convirtió en `meses_mora` recortando los negativos a 0 (`clip(lower=0)`).
4. `default payment next month` (0/1) se convirtió a `incumplio_pago` (`no`/`si`).
5. Muestra aleatoria de **600 filas** con `random_state=42`, y `id_cliente` reasignado como
   `C-0001 … C-0600`.

Se **descartaron deliberadamente** `SEX` y `MARRIAGE`: son atributos sensibles cuyo tratamiento
(sesgo, equidad, variables protegidas) excede el alcance de la clase 1 y merece su propia discusión.
También se descartaron los meses 2 a 6 de facturación, pago y estado (`BILL_AMT2-6`, `PAY_AMT2-6`,
`PAY_2-6`) para mantener el archivo chico y legible en una primera clase.

### Diccionario de datos

| Columna | Tipo | Descripción | Columna original |
|:--|:--|:--|:--|
| `id_cliente` | texto | Identificador del cliente (`C-0001` … `C-0600`), reasignado | `ID` |
| `edad` | entero | Edad del cliente en años (22–70 en esta muestra) | `AGE` |
| `nivel_educacion` | categórica | `posgrado`, `universitaria`, `media`, `otra` | `EDUCATION` |
| `limite_credito` | entero | Cupo de crédito aprobado | `LIMIT_BAL` |
| `monto_facturado_mes` | entero | Monto facturado en el mes. **Puede ser negativo**: indica saldo a favor del cliente | `BILL_AMT1` |
| `monto_pagado_mes` | entero | Monto efectivamente pagado en el mes | `PAY_AMT1` |
| `meses_mora` | entero | Meses de atraso en el pago (0 = al día; máximo 5 en esta muestra) | `PAY_0` |
| `incumplio_pago` | categórica | `si` / `no`: el cliente incumplió el pago del **mes siguiente** | `default payment next month` |

**Unidad monetaria:** los montos están en **dólares taiwaneses (NT$)**, la moneda del registro
original. No se convirtieron ni se reetiquetaron a pesos: el orden de magnitud es lo que importa
para el uso que se le da en clase.

**Distribución del objetivo:** 463 `no` y 137 `si` (22,8 % de incumplimiento). Es un dataset
**desbalanceado**, como casi todo problema real de morosidad.

---

## `absentismo_laboral.csv`

Registros de **ausencias del personal** de una empresa de mensajería, con los atributos del
empleado y del período en que ocurrió cada ausencia. Se usa en la clase 2 para **formular** el
problema antes de modelar: decidir qué representa una fila, qué se predice, con cuánta
anticipación y con qué variables se cuenta en el momento de decidir.

**Advertencia de uso:** este archivo conserva a propósito variables de cuerpo y hábitos
(`peso_kg`, `altura_cm`, `indice_masa_corporal`, `bebedor_social`, `fumador_social`). **No están
para usarse como predictoras**: están para que decidir si entran o no sea un ejercicio explícito
de la ficha de formulación. Ver «Sobre las variables sensibles», más abajo.

### Origen y licencia

- **Fuente:** UCI Machine Learning Repository, *Absenteeism at Work* (dataset 445) —
  <https://archive.ics.uci.edu/dataset/445/absenteeism+at+work>
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0) —
  <https://creativecommons.org/licenses/by/4.0/>
- **Obra derivada:** versión **modificada** del original. Se conservaron las 740 filas y las 21
  columnas, traducidas al español y con las categóricas recodificadas de números a texto, y se
  **agregó** una columna `periodo` reconstruida. El detalle está en «Preparación aplicada».
- **Cita original:** Martiniano, A., & Ferreira, R. (2018). *Absenteeism at work* [Dataset]. UCI
  Machine Learning Repository. Datos recogidos entre julio de 2007 y julio de 2010 en una empresa
  de mensajería de Brasil.
- **Datos reales y pseudonimizados**: no contiene nombres; `id_empleado` se reformateó a
  `E-01 … E-36` a partir del identificador numérico original.

### Preparación aplicada (reproducible)

El script que la genera está en `_prep_absentismo_laboral.py`, junto a este README. Es
autocontenido: descarga el original desde UCI, verifica sus supuestos y escribe el CSV.

1. Las 21 columnas se renombraron al español en `snake_case` (equivalencias en el diccionario).
2. Categóricas recodificadas de número a texto: `motivo_ausencia` (28 categorías, los códigos 1 a
   21 son capítulos de la CIE-10 y del 22 al 28 son motivos administrativos), `dia_semana`,
   `estacion`, `nivel_educacion`, y los indicadores `si`/`no`.
3. **`periodo` reconstruido** (columna nueva; ver abajo).
4. No se tomó submuestra ni se descartaron columnas: el archivo ya es chico.

#### La reconstrucción del período

El original **no trae año**, solo `Month of absence` de 1 a 12, así que a primera vista no hay
línea de tiempo — y sin ella no se pueden formular el horizonte ni el momento de la predicción.
Sí la hay, escondida en una columna de otra cosa: `Work load Average/day` es una cifra **mensual
de planta**, igual para todos los empleados del mismo mes. Se verificó que:

1. cada uno de sus 38 valores distintos corresponde a **un solo** mes calendario;
2. sus filas aparecen en **bloques contiguos que no se solapan** en el orden del archivo;
3. la secuencia de meses que implican esos bloques es **corrida, sin saltos**, de julio a julio.

Las tres condiciones se cumplen, de modo que el archivo está ordenado cronológicamente y el
bloque n-ésimo es el mes n-ésimo desde **2007-07**: **37 meses, de 2007-07 a 2010-07**. El script
**comprueba las tres con `assert` y aborta** si alguna deja de cumplirse.

### Diccionario de datos

| Columna | Tipo | Descripción | Columna original |
|:--|:--|:--|:--|
| `id_empleado` | texto | Identificador del empleado (`E-01` … `E-36`) | `ID` |
| `periodo` | texto | Año-mes de la ausencia (`2007-07` … `2010-07`). **Reconstruido**, vacío en las 3 filas sin mes | — (nueva) |
| `mes` | entero | Mes de la ausencia, 1–12; `0` en 3 filas sin registrar | `Month of absence` |
| `dia_semana` | categórica | `lunes` … `viernes` | `Day of the week` |
| `estacion` | categórica | `verano`, `otonio`, `invierno`, `primavera` (**hemisferio sur**) | `Seasons` |
| `motivo_ausencia` | categórica | 28 categorías; se conoce **después** de la ausencia | `Reason for absence` |
| `horas_ausencia` | entero | Horas de ausencia del episodio (0–120) | `Absenteeism time in hours` |
| `edad` | entero | Edad del empleado en años | `Age` |
| `antiguedad_anios` | entero | Años de servicio en la empresa | `Service time` |
| `nivel_educacion` | categórica | `media`, `universitaria`, `posgrado`, `magister o doctorado` | `Education` |
| `hijos` | entero | Cantidad de hijos | `Son` |
| `mascotas` | entero | Cantidad de mascotas | `Pet` |
| `distancia_casa_trabajo_km` | entero | Distancia entre residencia y trabajo | `Distance from Residence to Work` |
| `gasto_transporte` | entero | Gasto de transporte del empleado | `Transportation expense` |
| `carga_trabajo_promedio_dia` | decimal | Carga de trabajo promedio diaria **de la planta** ese mes | `Work load Average/day` |
| `cumplimiento_meta_pct` | entero | Cumplimiento de la meta del mes, en porcentaje | `Hit target` |
| `falta_disciplinaria` | categórica | `si` / `no` | `Disciplinary failure` |
| `bebedor_social` | categórica | `si` / `no`. **Variable sensible** | `Social drinker` |
| `fumador_social` | categórica | `si` / `no`. **Variable sensible** | `Social smoker` |
| `peso_kg` | entero | Peso del empleado. **Variable sensible** | `Weight` |
| `altura_cm` | entero | Estatura del empleado. **Variable sensible** | `Height` |
| `indice_masa_corporal` | entero | IMC del empleado. **Variable sensible** | `Body mass index` |

### Particularidades que conviene conocer antes de usarlo

- **Una fila NO es un empleado.** Son **36 empleados en 740 filas**: cada fila es un *episodio de
  ausencia*. La mediana es de 9 registros por empleado y el máximo, 113.
- **Una fila tampoco es siempre una ausencia.** Las **40 filas con `falta_disciplinaria = si`**
  tienen sin excepción `horas_ausencia = 0` y `motivo_ausencia = sin motivo registrado`: son
  registros disciplinarios mezclados en una tabla de ausencias, no episodios de ausencia.
- **Los meses sin ausencias no existen en el archivo.** Sobre la grilla completa de empleado ×
  período (33 empleados con ausencia real fechada × 37 meses = **1.221** filas), solo **366** están
  en los datos: el **70 %** restante son meses en que no pasó nada y que nadie registró. Son
  observaciones válidas, no datos faltantes, y omitirlas cambia el resultado (ver abajo).
- **Tres empleados quedan fuera de la tabla de análisis** al aplicar la definición del objetivo
  (con período y sin registros disciplinarios): `E-04` y `E-35` aparecen solo en las filas con
  `mes = 0`, y `E-08` solo en una de esas y en una falta disciplinaria. Ninguno tiene una ausencia
  real fechada, así que la grilla se arma sobre **33** empleados, no sobre 36.
- **34 duplicados exactos** y **cero nulos**.
- El **motivo de la ausencia se conoce después** de que ocurre. No está disponible en el momento
  en que hay que decidir, así que no puede usarse para predecir.

**Efecto de la unidad de observación sobre el resultado**, calculado sobre este archivo: un
baseline que predice siempre el promedio de horas por empleado-mes acierta muy distinto según qué
filas se consideren.

| Unidad de observación | Predicción del baseline | Error promedio |
|:--|--:|--:|
| Grilla completa (1.221 filas, meses sin ausencia como cero) | 4,20 h | 6,2 h |
| Solo las filas presentes en el archivo (366) | 14,00 h | 12,1 h |

Más del triple de diferencia, producida solo por una decisión de formulación.

### Sobre las variables sensibles

El original incluye peso, estatura, índice de masa corporal y consumo de alcohol y tabaco de cada
empleado. **Se conservaron a propósito**, apartándose del criterio con que
`morosidad_cartera.csv` descartó sexo y estado civil.

El motivo es pedagógico: la clase 2 trata de qué variables entran en la ficha de formulación, y
una variable que está disponible, que probablemente predice y que aun así no es defendible es el
mejor ejercicio posible para ese tema. Quitarla del archivo elimina la lección y no protege a
nadie, porque el original sigue publicado en UCI con esas columnas.

La distinción que el material debe dejar instalada es que **estar en la tabla** y **ser legítima
como predictora** son dos cosas distintas. Hay dos criterios de descarte y no conviene mezclarlos:
una variable se descarta por **no estar disponible a tiempo** (es el caso de `motivo_ausencia`) o
por **no ser legítima estándolo** (es el caso de estas cinco).

---

## `cobranza_servicios.csv`

Cartera de **facturas emitidas** por una empresa de servicios, con los datos del cliente, del
documento y de su cobranza. Se usa en la clase 3 para **auditar la calidad de un archivo**: cada
concepto de esa clase tiene acá un defecto concreto que se descubre con una técnica.

Es además el caso que los propios estudiantes formularon en la práctica autónoma de la clase 2
—anticipar qué facturas se van a pagar con atraso, para priorizar la gestión de cobranza—, así que
llegan a la clase 3 con la ficha ya hecha.

### Origen

- **Dato sintético**, generado para el curso. **No corresponde a ninguna empresa real** y así hay que
  presentarlo en clase.
- Los RUT son inventados, con **dígito verificador válido** para que se vean creíbles y para que el
  formato sea el que la audiencia reconoce.
- El script que lo genera es `_prep_cobranza_servicios.py`, junto a este README. Es **determinista**
  —semilla fija, sin fechas del sistema—, así que correrlo dos veces produce el mismo archivo, y
  **falla cerrado**: comprueba cada defecto que la clase afirma y, si alguno no se cumple, **no escribe
  el CSV** y termina con código de error. Es preferible quedarse sin archivo que publicar uno que
  contradiga el material.

### Por qué no se reutilizó el archivo de la clase 2

La clase 3 se armó primero sobre `absentismo_laboral.csv` y no funcionó. Cada hallazgo exigía
contexto que no estaba en los datos: las estaciones del hemisferio sur, que ciertas cifras eran de
planta y no del empleado, los códigos CIE-10, y un cero que unas veces era una ausencia de cero horas
y otras un registro disciplinario. Para una clase de **auditoría** eso está al revés: el defecto tiene
que descubrirse con una técnica, no porque el docente explicó antes el dominio.

En la clase 2 esa complejidad **era la lección** —la unidad de observación—. En la clase 3 era un
impuesto sobre cada hallazgo.

### Diccionario de datos

| Columna | Tipo | Descripción |
|:--|:--|:--|
| `id_factura` | texto | Identificador del documento (`F-0001`…). **No es predictora** |
| `rut_cliente` | texto | RUT del cliente, con puntos y dígito verificador |
| `razon_social` | texto | Nombre del cliente. **Ojo**: no siempre se escribe igual para el mismo RUT |
| `segmento` | categórica nominal | `pyme`, `corporativo`, `gobierno` |
| `region` | categórica nominal | Región del cliente |
| `ejecutivo` | categórica nominal | Ejecutivo de cuenta a cargo |
| `prioridad_cobranza` | categórica **ordinal** | `baja` < `media` < `alta` |
| `fecha_emision` | temporal | Fecha de emisión de la factura |
| `fecha_vencimiento` | temporal | Emisión más el plazo de crédito |
| `fecha_pago` | temporal | Fecha de pago efectivo |
| `dias_credito` | entero | Plazo pactado: 30, 60 o 90. **Es numérica pero funciona como categórica** |
| `monto_neto` | entero | Monto neto en pesos |
| `monto_total` | entero | Monto con IVA. **Derivado**: es el neto por 1,19 |
| `tasa_morosidad_cartera` | texto | Morosidad de **toda la cartera** ese mes, en porcentaje. Llega con **coma decimal**, así que se lee como texto |
| `estado_cobranza` | categórica | `pagada al día`, `pagada con atraso`, `en gestión judicial` |
| `monto_gestion_cobranza` | entero | Gasto de gestión de cobranza; **es 0 si la factura se pagó al día** |
| `dias_atraso` | entero | **Variable objetivo**: días entre el vencimiento y el pago |

### Los defectos, y qué concepto enseña cada uno

Todos verificados por el script al generar el archivo.

| Defecto | Cantidad | Concepto |
|:--|:--|:--|
| `dias_credito` es numérica y en realidad es categórica; `id_factura` no es predictora | — | 1 · Tipos |
| `tasa_morosidad_cartera` llega como **texto** por la coma decimal | 1 columna | 1 · Tipos |
| `monto_neto` fuertemente asimétrico: media $1.014.250 contra mediana $616.000 | — | 2 · Univariado |
| `monto_total` es **exactamente** `monto_neto × 1,19`: correlación **1,000** | — | 3 · Redundancia |
| Filas **duplicadas exactas** | 12 | 5 · Duplicados |
| RUT con **más de una razón social** | 5 | 5 · Inconsistencias |
| Facturas con **fecha de pago anterior a la de emisión** | 5 | 5 · Inconsistencias |
| Los defectos se **concentran en un ejecutivo**: las 24 filas involucradas en duplicados son todas suyas | — | 5 · Auditoría por rebanadas |
| Un contrato de **148 millones**: atípico **real**, no error | 1 | 6 · Atípicos |
| Montos cargados **en miles** en vez de pesos | 8 | 6 · Atípico que sí es error |
| `estado_cobranza` **se conoce después** del hecho que se quiere predecir | — | 7 · Fuga |
| `monto_gestion_cobranza` vale 0 **siempre** que no hubo atraso: casi determina la respuesta | — | 7 · Fuga |
| `tasa_morosidad_cartera` es **constante dentro de cada mes**: es del período, no de la factura | — | 7 · Fuga |

### Distribución del objetivo

El 33 % de las facturas se paga al día. Del resto, el atraso llega hasta 49 días. Por segmento:
corporativo 4,0 días en promedio, pyme 9,2 y gobierno 25,2 — una diferencia grande, interpretable y
accionable, que es lo que el análisis bivariado tiene que hacer visible.

Pero el promedio del segmento **no** es un buen criterio de acción: mirando el atraso promedio de cada
cliente, **26 de los 48 clientes pyme** se atrasan más que el cliente de gobierno que mejor paga. Se
cobra a clientes, no a segmentos.
