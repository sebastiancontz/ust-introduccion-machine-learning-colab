# Datasets · Introducción a Machine Learning (edición 2026)

Cada dataset declara su **origen** (de uso público con fuente y licencia, o dato sintético generado
para el curso) y en qué clase se usa. Las columnas están **en español y en `snake_case`**; cuando el
dataset original viene en inglés, acá queda documentada la equivalencia.

| Dataset | Registros × columnas | Origen | Usado en |
|:--|:--:|:--|:--|
| `morosidad_cartera.csv` | 600 × 8 | Público — UCI ML Repository, *Default of Credit Card Clients* (CC BY 4.0) | Clase 1 |
| `absentismo_laboral.csv` | 740 × 22 | Público — UCI ML Repository, *Absenteeism at Work* (CC BY 4.0) | Clase 2 |
| `cobranza_servicios.csv` | 612 × 17 | **Sintético**, generado para el curso | Clases 3 y 6 |
| `ventas_online.parquet` | 541.910 × 8 | Público — UCI ML Repository, *Online Retail II* (CC BY 4.0) | Clase 4 |
| `ventas_online_limpio.parquet` | 536.642 × 14 | Derivado del anterior: sus siete decisiones de limpieza de la clase 4 (CC BY 4.0) | Clases 5 y 6 |

El chequeo identifica un dato sintético por la palabra **Sintético** en su fila. Para datos de
terceros, documentar fuente y licencia efectivas. Para un dato propio sintético, escribir
**Sintético**, enlazar el script generador `.py` y declarar que no representa una organización real.
Si el titular no ha declarado una licencia para ese dato, indicar `No declarada`; no inventar una
condición legal para completar la tabla.

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
  600 registros y 8 de las 25 columnas, con las columnas traducidas al español y algunas
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
5. Muestra aleatoria de **600 registros** con `random_state=42`, y `id_cliente` reasignado como
   `C-0001 … C-0600`.

Se **descartaron deliberadamente** `SEX` y `MARRIAGE`: son variables sensibles cuyo tratamiento
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

Registros de **ausencias del personal** de una empresa de mensajería, con las variables del
empleado y del período en que ocurrió cada ausencia. Se usa en la clase 2 para **formular** el
problema antes de modelar: decidir qué representa un registro, qué se predice, con cuánta
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
- **Obra derivada:** versión **modificada** del original. Se conservaron los 740 registros y las 21
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
| `periodo` | texto | Año-mes de la ausencia (`2007-07` … `2010-07`). **Reconstruido**, vacío en los 3 registros sin mes | — (nueva) |
| `mes` | entero | Mes de la ausencia, 1–12; `0` en 3 registros sin registrar | `Month of absence` |
| `dia_semana` | categórica | `lunes` … `viernes` | `Day of the week` |
| `estacion` | categórica | `verano`, `otonio`, `invierno`, `primavera`. **Contradice a `mes`**: ver «Particularidades» | `Seasons` |
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

- **Un registro NO es un empleado.** Son **36 empleados en 740 registros**: cada registro es un *episodio de
  ausencia*. La mediana es de 9 registros por empleado y el máximo, 113.
- **Un registro tampoco es siempre una ausencia.** Los **40 registros con `falta_disciplinaria = si`**
  tienen sin excepción `horas_ausencia = 0` y `motivo_ausencia = sin motivo registrado`: son
  registros disciplinarios mezclados en una tabla de ausencias, no episodios de ausencia.
- **Los meses sin ausencias no existen en el archivo.** Sobre la grilla completa de empleado ×
  período (33 empleados con ausencia real fechada × 37 meses = **1.221** registros), solo **366** están
  en los datos: el **70 %** restante son meses en que no pasó nada y que nadie registró. Son
  observaciones válidas, no datos faltantes, y omitirlas cambia el resultado (ver abajo).
- **Tres empleados quedan fuera de la tabla de análisis** al aplicar la definición del objetivo
  (con período y sin registros disciplinarios): `E-04` y `E-35` aparecen solo en los registros con
  `mes = 0`, y `E-08` solo en una de esas y en una falta disciplinaria. Ninguno tiene una ausencia
  real fechada, así que la grilla se arma sobre **33** empleados, no sobre 36.
- **`estacion` contradice a `mes`, y no es un error de nuestra preparación.** Las etiquetas se
  conservan tal como salen de los códigos que UCI documenta (1 = *summer*, 2 = *autumn*,
  3 = *winter*, 4 = *spring*), pero los meses que agrupan no calzan con ninguno de los dos
  hemisferios. Verificado sobre el archivo con `crosstab(estacion, mes)`: «verano» son julio, agosto
  y septiembre; «otoño», enero, febrero y marzo; «invierno», abril, mayo y junio; y solo
  «primavera» (octubre, noviembre, diciembre) coincidiría con el hemisferio sur. Se probaron los dos
  hemisferios y ninguno hace calzar las cuatro. La contradicción viene del dataset original —sus
  datos no calzan con su propia documentación— y se detecta cruzando las dos columnas. Es la única
  inconsistencia **entre dos columnas** que tiene el archivo.
- **34 duplicados exactos** y **cero nulos**.
- El **motivo de la ausencia se conoce después** de que ocurre. No está disponible en el momento
  en que hay que decidir, así que no puede usarse para predecir.

**Efecto de la unidad de observación sobre el resultado**, calculado sobre este archivo: un
baseline que predice siempre el promedio de horas por empleado-mes acierta muy distinto según qué
registros se consideren.

| Unidad de observación | Predicción del baseline | Error promedio |
|:--|--:|--:|
| Grilla completa (1.221 registros, meses sin ausencia como cero) | 4,20 h | 6,2 h |
| Solo los registros presentes en el archivo (366) | 14,00 h | 12,1 h |

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
| Registros **duplicados exactos** | 12 | 5 · Duplicados |
| RUT con **más de una razón social** | 5 | 5 · Inconsistencias |
| Facturas con **fecha de pago anterior a la de emisión** | 5 | 5 · Inconsistencias |
| Los defectos se **concentran en un ejecutivo**: los 24 registros involucrados en duplicados son todos suyos | — | 5 · Auditoría por rebanadas |
| Un contrato de **148 millones**: atípico **real**, no error | 1 | 6 · Atípicos |
| Montos cargados **en miles** en vez de pesos | 8 | 6 · Atípico que sí es error |
| `estado_cobranza` **se conoce después** del hecho que se quiere predecir | — | 7 · Fuga |
| `monto_gestion_cobranza` vale 0 **siempre** que no hubo atraso: casi determina la respuesta | — | 7 · Fuga |
| `tasa_morosidad_cartera` es **constante dentro de cada mes**: es del período, no de la factura | — | 7 · Fuga |

### Distribución del objetivo

El 33 % de las facturas se paga al día. Del resto, el atraso llega hasta 49 días. Por segmento:
corporativo 4,0 días en promedio, pyme 9,2 y gobierno 25,2 — una diferencia grande, interpretable y
accionable, que es lo que el análisis bivariado tiene que hacer visible.

Pero el atraso promedio **no** es un buen criterio de acción, y el motivo no es que el promedio falle:
es que la métrica elegida no refleja la decisión. Priorizando por **exposición** —monto por días, o sea
cuánto dinero está detenido y por cuánto tiempo— gobierno y pyme quedan casi empatados, **49,1 %**
contra **42,8 %**, porque pyme son 364 facturas contra 85. Elegir la métrica que está a mano en vez de
la que refleja la decisión cambia a quién se llama.

> **Cifra retirada, para que nadie la reponga.** Una versión anterior de este README afirmaba que
> «26 de los 48 clientes pyme se atrasan más que el cliente de gobierno que mejor paga». El argumento
> se descartó el 2026-08-02 por débil —comparaba contra el piso del grupo peor— y además la cifra es
> un **artefacto**: solo se reproduce agrupando por `(rut_cliente, segmento)`. Con un cliente = un RUT,
> el mejor cliente de gobierno está en **18,7 días** y **ninguna** de las 48 pyme lo supera
> (verificado el 2026-08-04).

---

## `ventas_online.parquet`

Un año de **líneas de factura** de un mayorista británico de artículos de regalo: qué se vendió, a
quién, cuándo, en qué cantidad y a qué precio. Se usa en la clase 4 para **decidir qué hacer con un
archivo sucio**: cada concepto de esa clase tiene acá un defecto real sobre el que hay que
pronunciarse.

**Es un archivo real y NO está limpio.** Los defectos no se inyectaron: estaban ahí. Ese es
justamente el punto.

### Origen y licencia

- **Fuente:** UCI Machine Learning Repository, *Online Retail II* (dataset 502) —
  <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0) —
  <https://creativecommons.org/licenses/by/4.0/>
- **Cita original:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5CG6D>
- **Obra derivada:** versión **modificada** del original. Se tomó la hoja `Year 2010-2011` completa
  (541.910 registros, del 2010-12-01 al 2011-12-09) y se **renombraron las ocho columnas al español**. No
  se tocó nada más.

### Preparación aplicada (reproducible)

El script está en `_prep_ventas_online.py`, junto a este README. Es autocontenido: descarga el
original desde UCI, renombra las columnas y escribe el Parquet.

1. Las 8 columnas se renombraron al español en `snake_case` (equivalencias en el diccionario).
2. **No se limpió nada.** Ni faltantes, ni duplicados, ni atípicos, ni categorías, ni tipos.
3. **Los valores no se traducen.** Traducirlos convertiría un dato real en uno fabricado y además
   borraría defectos: `EIRE` y `RSA` conviven con nombres de país completos, y `Unspecified` es un
   faltante disfrazado de categoría. Las anotaciones que importan son cortas y se glosan en clase.
4. No se tomó submuestra: el archivo entra completo.

El script **falla cerrado**. Comprueba con `assert` cada defecto que el material de la clase afirma
y, si alguno no se cumple, **no escribe el archivo** y termina con código de error. Es preferible
quedarse sin dataset que publicar uno que contradiga las slides.

### Por qué Parquet y no CSV

- **El tipo de cada columna viaja guardado**, así que el archivo no depende de que quien lo lea
  acierte con los argumentos de lectura. Un CSV de estas dimensiones es frágil justamente donde esta
  clase no quiere fragilidad.
- **Pesa 3,2 MB contra 48 MB** del CSV equivalente, con los 136.534 faltantes y los tipos intactos.
- Se lee igual de simple, y **sobre una URL `https` funciona solo con pandas y `pyarrow`**, sin
  dependencias extra:

  ```python
  import pandas as pd
  ventas = pd.read_parquet(BASE + "ventas_online.parquet")
  ```

### Diccionario de datos

| Columna | Tipo | Descripción | Columna original |
|:--|:--|:--|:--|
| `n_factura` | texto | Número del documento. **Codifica dos hechos**: el prefijo `C` marca una venta cancelada | `Invoice` |
| `codigo_producto` | texto | Código del artículo (5 dígitos). **Ojo**: también guarda cosas que no son productos | `StockCode` |
| `descripcion_producto` | texto | Nombre del artículo. **Ojo**: también guarda anotaciones a mano del bodeguero | `Description` |
| `cantidad` | entero | Unidades de la línea. **Negativa** en las devoluciones y en algunos ajustes | `Quantity` |
| `fecha_factura` | temporal | Fecha y hora de emisión | `InvoiceDate` |
| `precio_unitario` | decimal | Precio por unidad, en libras esterlinas | `Price` |
| `id_cliente` | decimal | Identificador del cliente. **Es un identificador guardado como número, y no es predictor** | `Customer ID` |
| `pais` | categórica nominal | País del cliente. 38 valores, incluidos `Unspecified` y `European Community` | `Country` |

### Los defectos, y qué concepto enseña cada uno

Todos verificados por el script al generar el archivo (2026-08-04).

| Defecto | Cantidad | Concepto |
|:--|:--|:--|
| `id_cliente` es un identificador guardado como decimal (`17850.0`) | 1 columna | 1 · Tipos |
| `n_factura` codifica **dos hechos** en una columna: el número y si fue cancelada | 1 columna | 1 · Formato |
| `codigo_producto` mezcla productos con `POST`, `DOT`, `M`, `D`, `BANK CHARGES`, `AMAZONFEE`, `CRUK` | 1.257 + 710 + 571 + … | 1 · Formato |
| `descripcion_producto` guarda anotaciones a mano: `check`, `damages`, `damaged`, `?`, `Found`/`found` | 650 de 4.070 códigos tienen más de una descripción | 1 · Formato |
| **El mismo incidente desde los dos lados**: en el registro del código 23343 alguien escribió `20713` —otro código— en el campo del nombre, y los registros del 20713 lo explican con `Marked as 23343` y `wrongly coded 23343` | 1 + 2 registros | 1 · Formato |
| `pais` incluye `Unspecified`, que es un faltante disfrazado de categoría | 507 registros | 1 · Formato |
| `id_cliente` vacío | 135.080 · **24,9 %** | 2 · Faltantes |
| **La ausencia es informativa**: cantidad media 12,1 con cliente contra **2,0** sin cliente; ticket 20,40 contra 10,72; y 2.475 de los 2.515 registros con precio cero no tienen cliente | — | 2 · Ausencia informativa |
| `descripcion_producto` vacía — el mecanismo opuesto: pocos y recuperables desde el código | 1.454 · 0,27 % | 2 · Faltantes |
| Filas **duplicadas exactas** | 5.268 · 0,97 % | 4 · Duplicados |
| **El par que NO se borra**: factura `581483` con **+80.995** unidades de `PAPER CRAFT , LITTLE BIRDIE` a 2,08 y su cancelación `C581484` con **−80.995**, mismo cliente | 2 registros | 5 · Atípico real |
| **El que no es una venta**: `556690`, descripción `printing smudges/thrown away`, **−9.600** unidades, precio 0, sin cliente. Es un ajuste de inventario cargado en una tabla de ventas | 1 registro | 5 · Atípico que sí es error |
| Los registros con prefijo `C` tienen **todos** cantidad negativa (0 excepciones)… pero hay negativos **sin** prefijo | 9.288 · contra 1.336 | 5 · Inconsistencia |
| Registros con precio cero, y con precio negativo | 2.515 · y 2 | 5 · Atípicos |
| Si el problema es anticipar cancelaciones, el prefijo `C` y la cantidad negativa **son** la respuesta escrita de otra forma | — | 6 · Circularidad |

### Lo que SÍ está bien, y también se documenta

Una auditoría registra lo que está en orden, no solo lo que falla. En este archivo: `fecha_factura`
llega como fecha y `cantidad` como entero, el año está corrido y sin vacíos, y todo registro cancelado
tiene cantidad negativa sin una sola excepción.

### Escala

**541.910 registros está fuera del terreno cómodo de una planilla de cálculo** —Excel topa en 1.048.576
y sufre mucho antes—. Conviene decirlo en clase: es el argumento de por qué esta audiencia está en
este curso.

---

## `ventas_online_limpio.parquet`

El **resultado** de aplicar a `ventas_online.parquet` las siete decisiones de limpieza de la clase 4.
Es el archivo de **entrada de la clase 5**: allá no se corrige nada más, se transforma lo que ya está
correcto.

**Es *UNA* versión limpia, nunca *LA* versión correcta.** Dos de las siete decisiones —qué hacer con
el 24,9 % de `id_cliente` faltante y qué hacer con las cantidades negativas sin prefijo de
cancelación— admiten más de una respuesta defendible. Acá se eligió una y el generador declara cuál y
qué descartó. Un archivo canónico contradeciría en silencio la tesis de la clase 4, que es que no hay
reglas universales de limpieza.

### Origen y licencia

- **Creador:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5CG6D>
- **Fuente:** UCI Machine Learning Repository, *Online Retail II* (dataset 502) —
  <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0) —
  <https://creativecommons.org/licenses/by/4.0/>
- **Modificación:** obra derivada en **dos pasos**. Primero `_prep_ventas_online.py` renombró las
  ocho columnas al español sin tocar los valores; después
  [`_soluciones/_build_ventas_online_limpio.py`](../_soluciones/_build_ventas_online_limpio.py)
  aplicó las siete decisiones de limpieza que se detallan abajo, que eliminan 5.268 filas y agregan
  seis columnas.

### Qué le pasó al archivo

De **541.910 × 8** a **536.642 × 14**. Las siete decisiones, en el orden en que el generador las
aplica:

| # | Variable | Decisión | Efecto en el archivo |
|:--:|:--|:--|:--|
| 1 | `descripcion_producto` | completar **solo** desde códigos con un único nombre conocido | 1.033 registros recuperados; quedan 421 vacíos a propósito |
| 2 | `n_factura` | separar en `n_documento` y `cancelada`; se conserva la original | +2 columnas |
| 3 | (todas) | eliminar las repeticiones de **fila completa**, conservando la primera | −5.268 filas |
| 4 | `descripcion_producto` | normalizar a minúsculas y sin espacios sobrantes, en columna nueva | +1 columna (`descripcion_norm`) |
| 5 | `codigo_producto` | marcar lo que **no es un producto**; no se elimina | +1 columna (`es_producto`), 2.990 registros en `False` |
| 6 | `id_cliente` | conservar los registros y **marcar la ausencia**; no se imputa ni se excluye | +1 columna (`sin_cliente`), 135.037 registros en `True` |
| 7 | `cantidad` / `n_factura` | marcar los ajustes de inventario; no se eliminan | +1 columna (`ajuste_inventario`), 1.336 registros en `True` |

**Lo que NO se hizo, y es deliberado:** no se escaló, no se codificó ninguna categórica y no se creó
ninguna variable derivada. Eso es exactamente el trabajo de la clase 5.

### Diccionario de datos

Las ocho primeras columnas son las de `ventas_online.parquet` y su equivalencia con el original está
en el diccionario de ese archivo. Las seis últimas nacen acá.

| Columna | Tipo | Descripción |
|:--|:--|:--|
| `n_factura` | texto | Número del documento tal como venía, con el prefijo `C` cuando la venta se canceló |
| `codigo_producto` | texto | Código del artículo |
| `descripcion_producto` | texto | Nombre del artículo. **421 vacíos**: son los códigos con más de un nombre posible, que no se completaron por no elegir al azar |
| `cantidad` | entero | Unidades de la línea. **Negativa** en devoluciones y ajustes |
| `fecha_factura` | temporal | Fecha y hora de emisión |
| `precio_unitario` | decimal | Precio por unidad, en libras esterlinas |
| `id_cliente` | decimal | Identificador del cliente. **135.037 vacíos**, marcados por `sin_cliente`. No es predictor |
| `pais` | categórica nominal | País del cliente. 38 valores, incluidos `Unspecified` y `European Community` |
| `n_documento` | texto | **Nueva.** El número del documento sin el prefijo, para poder agrupar por factura |
| `cancelada` | booleana | **Nueva.** La venta se canceló. 9.251 registros en `True` |
| `descripcion_norm` | texto | **Nueva.** `descripcion_producto` en minúsculas y sin espacios sobrantes |
| `es_producto` | booleana | **Nueva.** La línea corresponde a un artículo y no a un cargo contable (`POST`, `DOT`, `BANK CHARGES`…). 533.652 registros en `True` |
| `sin_cliente` | booleana | **Nueva.** La venta no tiene cliente identificado. La ausencia es **informativa**: ese segmento se comporta distinto |
| `ajuste_inventario` | booleana | **Nueva.** Baja de inventario cargada en la tabla de ventas, no una venta. 1.336 registros en `True` |

### Qué representa cada registro, y por qué importa en la clase 5

Un registro sigue siendo una **línea de factura**: un producto dentro de una factura, no una venta
completa ni un cliente. La clase 5 **agrega a nivel de factura** para su práctica, y ese cambio de
unidad de observación se declara en voz alta: es una devolución a lo que la clase 2 enseñó.

### La bitácora no viaja con el archivo

El **por qué** de cada decisión vive en `_soluciones/bitacora_limpieza.csv`, que no se publica. El
archivo dice qué cambió; la bitácora, por qué. La clase 5 necesita el resultado, no la justificación.
