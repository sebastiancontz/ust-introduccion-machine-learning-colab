#!/usr/bin/env python3
"""Construye notebooks/06-pipelines-fuga-datos.ipynb.

POR QUÉ EXISTE UN GENERADOR: el `.ipynb` no se edita a mano. Si hay que corregir algo, se corrige acá y
se vuelve a ejecutar. En la clase 1 se perdieron dos correcciones por parchear el artefacto.

ESTRUCTURA: los tres tiempos de la `practica` del temario, más el cierre. (1) El docente parte el archivo
en tres y arma el `Pipeline` con el `ColumnTransformer` de la clase 5 adentro. (2) Entre todos se
incorpora el precio de catálogo aprendido SOLO con entrenamiento, y aparecen las dos cosas que no estaban
en el guion. (3) Cada uno construye el precio medio de la factura y decide si la cifra es creíble.
Cierre: se abre la prueba UNA vez.

LA PARTICIÓN ES TRIPLE Y SE USA DE VERDAD (corrección de la auditoría de la Fase 1): 60 / 20 / 20. Las
tres comparaciones didácticas van contra VALIDACIÓN; la PRUEBA se abre una sola vez, al final, sobre la
representación ya elegida. La versión anterior de la práctica medía tres veces contra prueba, o sea que
ensayaba justo la conducta que la clase condena.

EL ENCUADRE SE DECLARA ANTES DE EMPEZAR, y sin él la actividad no se puede evaluar: objetivo, el modelo
como instrumento de medición heredado de la clase 5, y las fuentes permitidas. La fuga de la práctica es
de CIRCULARIDAD y se diagnostica por álgebra, no por cronología: acá no hay un momento de decidir porque
el modelo no predice facturas futuras.

CÓMO SE MIDE: la receta de la clase 5, íntegra. Modelo lineal sobre `log1p` del monto, predicción
destransformada con `expm1`, MEDIANA DEL ERROR ABSOLUTO EN LIBRAS. Lo único que cambia es DÓNDE se mide.
SIN R² ni métricas con nombre propio: son de la clase 8.

LA CIFRA NUNCA VIAJA SOLA: siempre el par entrenamiento / validación, y al final la de prueba junto a la
de validación que la precedía.

TODAS LAS CIFRAS SALEN DE `assets/ilustraciones/_build_c06_figuras.py`, que las imprime y las protege con
diez aserciones. Ninguna se escribe a mano acá; las que el notebook muestra las CALCULA y las imprime con
Markdown dinámico.

FRONTERAS, heredadas del `alcance_no` y verificadas al escribir:
  · No hay validación cruzada ni búsqueda de hiperparámetros: clase 13. Tampoco `GridSearchCV`.
  · No se guarda ni se despliega el flujo: clase 16.
  · No se nombra «sobreajuste» ni se explica su tratamiento: clase 8. La brecha se mide y se lee.
  · No se estratifica: el objetivo es una cantidad, no una clase. Se enuncia para la clase 9.
  · No se compara un modelo contra otro: se comparan REPRESENTACIONES con un único modelo.

Uso:
    conda run -n ust-ml python ediciones/2026/notebooks/_build_06_pipelines_fuga_datos.py
      (después: `make nb C=06` para validarlo, y `make nb C=06 KEEP_OUTPUT=...` para dejarlo con salidas)
"""
from __future__ import annotations

import ast
import base64
import json
import statistics
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "06-pipelines-fuga-datos.ipynb"
ILUSTRACIONES = REPO_ROOT / "assets" / "ilustraciones"


def figura_base64(nombre: str, alt: str, width: int = 900) -> str:
    """Diagrama embebido como data URI: el notebook queda autocontenido y no depende de Pages.

    `alt` es obligatorio: sin él la figura no existe para quien usa lector de pantalla.
    """
    ruta = ILUSTRACIONES / nombre
    b64 = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return (f'<p align="center"><img src="data:image/svg+xml;base64,{b64}" '
            f'width="{width}" alt="{alt}"></p>')


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def con_ids(celdas: list[dict]) -> list[dict]:
    """nbformat 4.5+ exige un id por celda. Deterministas: regenerar no ensucia el diff."""
    for i, celda in enumerate(celdas):
        celda["id"] = f"c{i:02d}"
    return celdas


CELDAS = [
    md("""
<div style="display:flex; align-items:center; gap:18px; text-align:left">
<img src="https://sebastiancontz.github.io/ust-introduccion-machine-learning/assets/logo-ust.svg" width="100" alt="Logo de la Universidad Santo Tomás">
<div>
<p>Ingeniería en Información y Control de Gestión</p>
<p>Facultad de Economía y Negocios</p>
<p>Introducción a Machine Learning</p>
<p>Semana 06: Pipelines y fuga de datos</p>
</div>
</div>
"""),
    md("""
# 06 · Pipelines y fuga de datos

La clase pasada terminamos con un modelo que se equivocaba en **71,2 libras** por factura. Esa cifra se
calculó sobre las mismas facturas con las que se ajustó el modelo, así que no dice cómo le irá con
facturas nuevas: dice cuán bien **reproduce** lo que ya vio.

Un aviso para más adelante: las cifras de hoy no son comparables una a una con ese 71,2, porque desde
ahora el modelo se ajusta con el 60 % del archivo y no con todo. Lo que se compara entre las dos clases
es **dónde** se mide, no el valor.

Hoy separamos una parte del archivo antes de tocar nada, y desde ese momento la cifra **viaja en par**.

## Qué vamos a hacer

1. Partir el archivo en **tres** conjuntos, con la semilla declarada.
2. Meter el `ColumnTransformer` de la clase 5 dentro de un **`Pipeline`** y sacar el par de cifras.
3. Incorporar al flujo la variable que ustedes crearon la clase pasada.
4. Construir una variable nueva y **decidir si la cifra que produce es creíble**.
5. Abrir el conjunto de prueba. **Una vez.**

## Criterio de éxito de la actividad

En el paso 4 van a construir una variable nueva y medirla. El criterio **no es cuánto baja el error**:
es que decidan si la cifra resultante es **creíble**, y que expliquen esa decisión mirando **de dónde
sale cada columna** en vez del número que produce.
"""),
    md("""
## El encuadre, antes de escribir una línea

Tres cosas quedan fijadas y no se cambian durante la actividad:

- **Objetivo:** el monto total de la factura.
- **El modelo lineal es un instrumento de medición**, no un predictor. Igual que en la clase 5, sirve
  para comparar dos representaciones de las mismas variables. Acá no se decide nada sobre una factura
  futura.
- **Fuentes permitidas:** las columnas del archivo, más lo que se pueda resumir del **histórico de otras
  facturas**. Queda prohibida cualquier columna calculada a partir del monto de la propia factura.

Ese tercer punto es el que vuelve verificable el criterio de éxito. Y fíjense en algo: acá **no hay un
momento de decidir** en el sentido cronológico, porque no estamos prediciendo facturas futuras. La
frontera de hoy es otra y es más limpia: una columna calculada desde el objetivo de su propia fila es
**circular**, y eso se demuestra con álgebra.
"""),
    md("""
## Preparación del entorno
"""),
    code("""
%%capture
!pip install -q pandas numpy pyarrow scikit-learn
"""),
    code("""
import os

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
"""),
    code("""
def num(x, dec=0):
    \"\"\"Miles con punto y decimales con coma, como en el resto del material del curso.\"\"\"
    return f'{x:,.{dec}f}'.replace(',', '@').replace('.', ',').replace('@', '.')

def con_coma(x):
    \"\"\"La versión que usan las tablas, para que no salgan con punto decimal.\"\"\"
    return num(x, 1)
"""),
    md("""
## Cargar datos

El mismo archivo de las clases 4 y 5: las ventas con las siete decisiones de limpieza ya aplicadas.
"""),
    code("""
REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

lineas = pd.read_parquet(BASE + 'ventas_online_limpio.parquet')
print('Filas y columnas:', lineas.shape)
"""),
    md("""
### La base analítica, declarada una vez

Antes del filtro, dos palabras que vamos a usar todo el día y que conviene tener claras, porque el
archivo **no** trae un registro por venta:

- Una **línea de factura** es **un producto dentro de una factura**. Si una factura lleva cinco
  artículos distintos, ocupan **cinco filas** con el mismo `n_documento`. Ése es el registro del archivo
  tal como llega: su unidad de observación.
- **De producto** distingue las líneas que corresponden a un artículo de las que son un **cargo
  contable** cargado en la misma tabla: envíos (`POST`), comisiones bancarias (`BANK CHARGES`),
  comisiones de plataforma (`AMAZONFEE`). La clase 4 las marcó con `es_producto` y **no** las borró
  —son hechos contables reales—, así que acá simplemente las dejamos fuera del análisis.

Con eso: nos quedamos con las líneas **de producto**, no canceladas, con cantidad y precio positivos.
**Todas** las cifras de hoy salen de este estado y de ningún otro.

<!-- contrato: accion=fijar-base -->
"""),
    code("""
lineas = lineas[
    (lineas["es_producto"]) & (~lineas["cancelada"]) & (lineas["cantidad"] > 0) & (lineas["precio_unitario"] > 0)
].copy()
lineas['monto_linea'] = lineas["cantidad"] * lineas["precio_unitario"]
display(Markdown(f'Quedan **{num(len(lineas))} líneas** de producto.'))
"""),
    md("""
Y agregamos por documento para tener la tabla por **factura**: ahí cada registro es una factura
completa, con todos sus productos sumados. Es el cambio de unidad de observación que trabajaron en la
clase 5.
"""),
    md(figura_base64(
        "c05-de-linea-a-factura.svg",
        "A la izquierda, cinco filas del archivo con el mismo número de documento repetido: son cinco "
        "líneas de una misma factura. A la derecha, esas cinco filas agrupadas en una sola, con el "
        "número de líneas, las unidades y el monto total. Al pie, 522.504 líneas de producto se "
        "convierten en 19.773 facturas.")),
    md("""
De acá en adelante, cuando digamos **factura** nos referimos a un registro de esta segunda tabla.

<!-- contrato: accion=agregar-por-factura -->
"""),
    code("""
facturas = lineas.groupby('n_documento').agg(
    monto_total=('monto_linea', 'sum'),
    n_lineas=('codigo_producto', 'size'),
    unidades=('cantidad', 'sum'),
    pais=('pais', 'first'),
    fecha=('fecha_factura', 'first'),
).reset_index()
facturas['mes'] = facturas["fecha"].dt.month
"""),
    code("""
display(Markdown(f'**{num(len(facturas))} facturas**, y la factura mediana vale '
                 f'**{num(facturas["monto_total"].median(), 1)} libras**.'))
"""),
    md("""
# 1 · Partir el archivo

Antes de tocar nada. Y no en dos partes, en **tres**, porque cada una tiene un trabajo distinto:

| Conjunto | Para qué sirve |
|:--|:--|
| **Entrenamiento** | el modelo aprende acá, y solo acá |
| **Validación** | acá comparamos alternativas y decidimos cuál queda |
| **Prueba** | se abre **una sola vez**, al final, sobre la decisión ya tomada |

Si comparáramos alternativas mirando la prueba, la cifra que reportamos al final ya estaría inflada:
habríamos elegido lo que mejor le va a ese conjunto en particular.

### `train_test_split`

Es la función de scikit-learn que sortea la partición. Parte en **dos**, así que para tener tres
conjuntos la llamamos dos veces: primero apartamos la prueba, después partimos el resto.

`random_state` es la **semilla** (*random seed*): el número de partida del sorteo. Con la misma semilla
el archivo se parte siempre igual, y sin ella dos personas con el mismo código obtienen cifras
distintas. La declaramos una vez y la reutilizamos.

**Los nombres son los de siempre**, y conviene acostumbrarse porque son los que van a encontrar en la
documentación de scikit-learn y en cualquier tutorial: `X` son las predictoras, `y` el objetivo, y cada
conjunto lleva su sufijo — `X_train`, `X_val`, `X_test` y sus `y_` correspondientes.
"""),
    code("""
CATEGORICAS = ['pais', 'mes']            # la única lista que hay que mantener a mano

X = facturas[['n_lineas', 'unidades'] + CATEGORICAS]
y = facturas["monto_total"]
"""),
    md("""
`train_test_split` parte en **dos**, así que se llama **dos veces**: primero se aparta la prueba, y
después se parte el resto en entrenamiento y validación.

<!-- contrato: accion=partir-en-tres -->
"""),
    code("""
from sklearn.model_selection import train_test_split

SEMILLA = 42

# primero se aparta la prueba; lo que queda son entrenamiento y validación todavía juntos
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)

# y ahora ese bloque se parte; el 0,25 es del 80 % que quedó, o sea el 20 % del archivo
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25,
                                                  random_state=SEMILLA)
"""),
    code("""
display(Markdown(f'Entrenamiento **{num(len(X_train))}** · validación **{num(len(X_val))}** '
                 f'· prueba **{num(len(X_test))}**'))
"""),
    md(figura_base64(
        "c06-muro-verbos.svg",
        "Una barra dividida en tres bloques proporcionales: entrenamiento con 11.863 facturas, "
        "validación con 3.955 y prueba con 3.955. Una línea vertical gruesa separa el entrenamiento del "
        "resto. A la izquierda del muro dice fit más transform, acá se aprenden los parámetros; a la "
        "derecha, solo transform, se aplican los ya aprendidos.")),
    md("""
# 2 · El flujo, con el preprocesador de la clase 5 adentro

El `ColumnTransformer` de la semana pasada entra **intacto**: se conserva qué transformación va en cada
columna. Lo que cambia es **cuándo** se calcula.

### `Pipeline`

Encadena una secuencia de transformadores y un estimador final en **un solo objeto**. Al ajustarlo, los
datos recorren cada paso en orden; al predecir, recorren los mismos pasos con los parámetros ya
aprendidos. Gana tres cosas:

1. El ajuste ocurre en **un solo lugar**, así que solo puede ocurrir de un lado del muro.
2. El orden de los pasos vive dentro del objeto, no en el orden en que ejecutamos las celdas.
3. El objeto completo entra entero a lo que venga después.

### `SimpleImputer`

Rellena los valores que faltan. Y lo que importa hoy es **de dónde saca el valor con que rellena**:

- Al **ajustarse**, calcula un número por cada columna numérica —acá, la mediana— y se lo guarda.
- Al **aplicarse**, usa esos números guardados. No vuelve a mirar los datos que le llegan.

O sea que es el mismo mecanismo del escalador de la clase 5, con otra cuenta: aprende un parámetro y
después lo aplica. Por eso vive **dentro** del flujo, donde el ajuste ocurre de un solo lado del muro.

**¿Y si ahora no falta nada?** Es cierto: las cuatro columnas que tenemos están completas, así que hoy
el imputador no hace nada. Está puesto para lo que viene — en un rato va a aparecer una fila sin dato, y
sin él el modelo directamente **no corre**: `LinearRegression` se detiene con un error apenas encuentra
un faltante.

**¿Por qué va antes del logaritmo?** Podría ir después y daría lo mismo, y la razón es la que la clase 5
ya usó para destransformar con `expm1`: la mediana sobrevive a una transformación monótona, así que el
logaritmo de la mediana y la mediana de los logaritmos coinciden. Va primero simplemente porque se lee
mejor: primero se completa el dato, después se transforma.

### Qué son los nombres entre paréntesis

En el código de abajo, `'rellenar'`, `'logaritmo'`, `'columnas numericas'` y `'columnas categoricas'`
**no significan nada para scikit-learn**: son etiquetas que elegimos nosotros. Podrían decir `'paso 1'`
y funcionaría igual.

Hacen falta porque las dos piezas reciben una **lista de pasos**, y cada paso va acompañado de su
nombre. La diferencia entre las dos está en cuántos datos lleva cada paso:

- En un **`Pipeline`**, cada paso es un par: **(nombre, qué hacer)**. Los pasos se aplican uno después
  del otro, en ese orden.
- En un **`ColumnTransformer`**, cada paso es un trío: **(nombre, qué hacer, a qué columnas)**. Acá el
  orden no importa, porque cada uno trabaja sobre columnas distintas y después se pegan los resultados.

¿Para qué sirven los nombres, entonces? Para leerlos. Cuando dibujemos el flujo en un rato, cada caja va
a aparecer rotulada con ellos, y ahí se agradece que digan algo.

### Por qué las numéricas se deducen y no se listan

Acuérdense de la clase 5: **una columna que el `ColumnTransformer` no nombra se cae**. Y hoy le vamos a
agregar columnas nuevas a `X` un par de veces.

Si la ruta numérica trabajara con una lista fija, cada columna que agregáramos quedaría fuera **en
silencio** —el flujo correría igual, y la cifra sería la misma que sin ella—. Así que la deducimos:
categóricas son las que declaramos, y numéricas es **todo lo demás**. Con eso hay una sola lista que
mantener, y es la que no se puede deducir.

La línea que lo hace se lee de izquierda a derecha: *toma cada nombre de columna de `X` y quédate con
los que no estén en `CATEGORICAS`*.
"""),
    code("""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
"""),
    code("""
def preprocesador(X):
    numericas = [c for c in X.columns if c not in CATEGORICAS]     # el resto de las columnas
    ruta_numerica = Pipeline([
        ('rellenar', SimpleImputer(strategy='median')),             # guarda una mediana por columna
        ('logaritmo', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
    ])
    return ColumnTransformer([
        ('columnas numericas', ruta_numerica, numericas),
        ('columnas categoricas', OneHotEncoder(handle_unknown='ignore'), CATEGORICAS),
    ])
"""),
    md("""
Y la función que mide. Es la **receta de la clase 5**, sin cambios: ajusta sobre el logaritmo del monto,
destransforma la predicción y reporta la **mediana del error absoluto en libras**.

Lo partimos en **dos funciones**, porque son dos cosas distintas y conviene verlas separadas: una
ajusta el flujo y la otra mide un conjunto cualquiera con el flujo ya ajustado.

<!-- contrato: accion=armar-el-flujo -->
"""),
    code("""
def ajustar(X_train, y_train):
    flujo = Pipeline([
        ('preparar', preprocesador(X_train)),
        ('modelo', LinearRegression()),
    ])
    flujo.fit(X_train, np.log1p(y_train))         # el único fit, y solo con entrenamiento
    return flujo
"""),
    code("""
def error(flujo, X, y):
    prediccion = np.expm1(flujo.predict(X))
    return float(np.median(np.abs(prediccion - y.values)))
"""),
    md("""
Ahora sí: el modelo de la clase 5, con sus cuatro columnas, medido a los **dos lados** del muro.
"""),
    code("""
flujo_base = ajustar(X_train, y_train)

error_train = error(flujo_base, X_train, y_train)
error_val = error(flujo_base, X_val, y_val)
"""),
    code("""
mediana_factura = facturas["monto_total"].median()

resumen = pd.DataFrame({'error mediano (libras)': [error_train, error_val]},
                       index=['entrenamiento', 'validación'])
resumen['% de la factura típica'] = 100 * resumen['error mediano (libras)'] / mediana_factura
resumen.style.format(con_coma)
"""),
    md("""
### ¿72 libras es mucho o poco?

La cifra no dice nada sola. Hace falta un **baseline**: qué error tendría el modelo más tonto posible, el
que no mira ninguna predictora y predice siempre lo mismo. Es el baseline de la clase 2, ahora con la
herramienta que scikit-learn trae para eso.

### `DummyRegressor`

Predice una constante e **ignora por completo** las predictoras. No sirve para predecir: sirve como
vara. Si el modelo no le gana con claridad, no está aprendiendo nada útil de las columnas.

Y fíjense en cómo entra: el flujo es **el mismo**, solo cambia el último paso. Eso es lo que hace que
este pipeline se pueda reutilizar toda la unidad siguiente cambiándole el estimador.
"""),
    code("""
from sklearn.dummy import DummyRegressor

baseline = Pipeline([
    ('preparar', preprocesador(X_train)),      # el mismo preparado; al dummy le da igual, y se ve
    ('modelo', DummyRegressor(strategy='median')),
])
baseline.fit(X_train, np.log1p(y_train))
"""),
    code("""
error_baseline = error(baseline, X_val, y_val)

# lo que el dummy predice, preguntándoselo a él: sale igual para cualquier factura
constante = np.expm1(baseline.predict(X_val.iloc[:1]))[0]
"""),
    code("""
display(Markdown(
    f'El baseline predice **{num(constante, 1)} libras** para todas las facturas y se equivoca en '
    f'**{num(error_baseline, 1)}**. El modelo de la clase se equivoca en **{num(error_val, 1)}**, o sea '
    f'{num(100 * (1 - error_val / error_baseline))} % menos.'))
"""),
    md("""
Ahora el 72 significa algo: el modelo corta a menos de la mitad el error de decir lo mismo de todas. Con
eso en la mano ya se puede mirar la otra pregunta.

### Antes de leer la diferencia

Las dos cifras se parecen. Pero **antes de interpretar esa diferencia hay que saber cuánto se mueve
sola**: si cambiamos solo la semilla, el archivo se parte de otra manera y las cifras cambian con él.

Veinte particiones del mismo archivo, con la misma receta y el mismo modelo.
"""),
    code("""
validaciones = []

for semilla in range(20):
    # el guion bajo es la prueba, que en este experimento no se usa
    X_trainval_semilla, _, y_trainval_semilla, _ = train_test_split(X, y, test_size=0.2, random_state=semilla)
    X_train_semilla, X_val_semilla, y_train_semilla, y_val_semilla = train_test_split(X_trainval_semilla, y_trainval_semilla, test_size=0.25, random_state=semilla)
    validaciones.append(error(ajustar(X_train_semilla, y_train_semilla), X_val_semilla, y_val_semilla))
"""),
    code("""
amplitud = max(validaciones) - min(validaciones)
brecha = error_val - error_train
"""),
    code("""
display(Markdown(
    f'La validación se mueve entre **{num(min(validaciones), 1)}** y **{num(max(validaciones), 1)}** '
    f'libras: una amplitud de **{num(amplitud, 1)}**.\\n\\n'
    f'La diferencia que medimos es de **{num(brecha, 1)}** libras: menos de lo que se mueve la cifra '
    f'sola, así que no alcanza para afirmar nada.'))
"""),
    md(figura_base64(
        "c06-una-y-veinte-particiones.svg",
        "Dos paneles con el mismo eje vertical en libras por factura. El panel izquierdo muestra un solo "
        "par de puntos: entrenamiento en 70,3 y validación en 72,3. El panel derecho muestra veinte "
        "pares: los de validación se dispersan entre 67,4 y 75,1, y en varias particiones el punto de "
        "validación queda por debajo del de entrenamiento.")),
    md("""
Guárdense esa amplitud, porque es la **baranda** de todo lo que sigue: una diferencia menor que eso no
alcanza para afirmar nada.

# 3 · Una columna que hay que aprender, no leer

Hasta acá todas las columnas venían en el archivo: se leen y se usan. La que sigue es distinta, y esa
diferencia es el tema del bloque.

El **precio de catálogo** de un producto es su mediana histórica: hay que calcularlo mirando muchas
líneas. O sea que no es un dato, es un **parámetro aprendido de los datos** — igual que la media que
usa un escalador o el valor de relleno de un imputador. Y a todo lo que se aprende le aplica la regla
de hoy: se aprende **solo con entrenamiento**.

Para pegarlo usamos `assign`, que devuelve una copia de la tabla **con una columna nueva** en vez de
modificar la original. Conviene justamente cuando queremos comparar dos versiones de los mismos datos,
como vamos a hacer enseguida.
"""),
    code("""
documentos_entrenamiento = set(facturas["n_documento"].loc[X_train.index])
catalogo = (lineas[lineas["n_documento"].isin(documentos_entrenamiento)]
            .groupby('codigo_producto')["precio_unitario"].median())
"""),
    code("""
precio_por_factura = (lineas.assign(precio=lineas["codigo_producto"].map(catalogo))
                      .groupby('n_documento')["precio"].mean())
precio_catalogo = facturas["n_documento"].map(precio_por_factura)      # alineado al índice de facturas
sin_dato = precio_catalogo.isna().sum()
"""),
    code("""
display(Markdown(f'Facturas sin precio de catálogo aprendido: **{sin_dato}** de '
                 f'{num(len(facturas))}.'))
"""),
    md("""
Ahí está la primera cosa que no estaba en el guion. **Una** factura se queda sin precio de catálogo,
porque su producto no aparece en ninguna factura de entrenamiento.

No es un error: es exactamente lo que pasa cuando llega algo que el modelo no vio. Y por eso el
imputador tiene que estar **dentro** del flujo — su valor de relleno se aprende, y se aprende de un solo
lado del muro.
"""),
    code("""
# `assign` alinea por índice, así que a cada parte le llega el precio de sus propias facturas
X_train_catalogo = X_train.assign(precio_catalogo_medio=precio_catalogo)
X_val_catalogo = X_val.assign(precio_catalogo_medio=precio_catalogo)
X_test_catalogo = X_test.assign(precio_catalogo_medio=precio_catalogo)
"""),
    code("""
flujo_catalogo = ajustar(X_train_catalogo, y_train)

error_train_catalogo = error(flujo_catalogo, X_train_catalogo, y_train)
error_val_catalogo = error(flujo_catalogo, X_val_catalogo, y_val)
"""),
    code("""
display(Markdown(f'Entrenamiento **{num(error_train_catalogo, 1)}** · '
                 f'validación **{num(error_val_catalogo, 1)}** libras.'))
"""),
    md("""
### Mirar adentro del flujo

Un `Pipeline` no es una caja cerrada, y conviene no tratarlo como si lo fuera. Podemos comprobar **qué
hizo con una fila concreta** sin abrirlo por dentro, con un truco simple.

La factura que quedó sin precio de catálogo tuvo que rellenarse con algo. Si ese algo es la mediana de
entrenamiento, entonces predecir esa factura **tal como llega** y predecirla **rellenada a mano** con esa
mediana tienen que dar exactamente lo mismo. Probémoslo.
"""),
    code("""
sin_precio = X_val_catalogo['precio_catalogo_medio'].isna()
factura_sin_precio = X_val_catalogo[sin_precio]
"""),
    code("""
# la misma factura, pero rellenada a mano con la mediana de entrenamiento
mediana_entrenamiento = X_train_catalogo['precio_catalogo_medio'].median()
rellenada_a_mano = factura_sin_precio.assign(precio_catalogo_medio=mediana_entrenamiento)
"""),
    code("""
sin_dato = np.expm1(flujo_catalogo.predict(factura_sin_precio))[0]
a_mano = np.expm1(flujo_catalogo.predict(rellenada_a_mano))[0]

display(Markdown(
    f'Predicción de la factura **tal como llega**, sin el dato: {num(sin_dato, 1)} libras.\\n\\n'
    f'Predicción **rellenando a mano** con {num(mediana_entrenamiento, 2)}: {num(a_mano, 1)} libras.'
    f'\\n\\n¿Son exactamente iguales? **{np.isclose(sin_dato, a_mano)}**'))
"""),
    md("""
Idénticas, que es lo que tenía que pasar: el flujo rellenó con la mediana que aprendió de
entrenamiento. La cifra en sí no importa —esa factura es rarísima—; lo que importa es que las dos
coincidan. Es la regla del día, ocurriendo donde se puede comprobar.

Y ojo con lo que **no** habría pasado si el imputador estuviera fuera del flujo: la habríamos rellenado
antes de partir, con la mediana de todo el archivo, y esta comprobación no tendría con qué compararse.

### La pregunta incómoda

¿Y si calculamos ese mismo catálogo sobre **todo** el archivo, en vez de solo con entrenamiento? El
parámetro incorporaría información de los conjuntos que deberían estar tapados: es fuga de
procedimiento, la primera fila del catálogo de formas.

Antes de correr la celda, arriesguen un número: ¿cuánto **cambia** la cifra?
"""),
    code("""
catalogo_global = lineas.groupby('codigo_producto')["precio_unitario"].median()
precio_global = (lineas.assign(precio=lineas["codigo_producto"].map(catalogo_global))
                 .groupby('n_documento')["precio"].mean())
precio_catalogo_global = facturas["n_documento"].map(precio_global)
"""),
    code("""
X_train_global = X_train.assign(precio_catalogo_medio=precio_catalogo_global)
X_val_global = X_val.assign(precio_catalogo_medio=precio_catalogo_global)

flujo_global = ajustar(X_train_global, y_train)
"""),
    code("""
error_train_global = error(flujo_global, X_train_global, y_train)
error_val_global = error(flujo_global, X_val_global, y_val)
"""),
    code("""
comparacion = pd.DataFrame(
    {'aprendido con entrenamiento': [error_train_catalogo, error_val_catalogo],
     'aprendido con todo el archivo': [error_train_global, error_val_global]},
    index=['entrenamiento', 'validación'])
comparacion.style.format(con_coma)
"""),
    code("""
diferencia = abs(error_val_global - error_val_catalogo)
display(Markdown(f'La diferencia en validación es de **{num(diferencia, 1)} libras**.'))
"""),
    md("""
Casi nada. Y conviene decirlo sin adornos: **sobre este archivo, romper la regla no se paga**. La
mediana de cada producto se calcula con más de medio millón de líneas, así que quitar el 40 % no la
mueve.

Eso no es un accidente nuestro. La bibliografía del curso lo dice: lo correcto es calcular esos
parámetros solo con entrenamiento, pero con conjuntos grandes la diferencia suele ser mínima, porque
las dos particiones tienen distribuciones parecidas. El riesgo, entonces, es leer la regla como
purismo estadístico.

### Por qué igual se cumple

Tres razones, y ninguna es la cifra.

**Primera: el tamaño no siempre salva.** Acá el parámetro se calcula con medio millón de líneas. Con
una columna que tenga pocos registros por categoría, o un imputador sobre una columna con muchos
vacíos, el mismo incumplimiento sí mueve el resultado. La regla tiene que valer antes de saber en qué
caso estamos.

**Segunda: hay parámetros que sí se contaminan con un solo dato.** Un escalamiento que use el máximo
del archivo completo es el ejemplo clásico: si ese máximo está en el conjunto de prueba, el modelo
queda informado del rango de datos que todavía no debería conocer. Este archivo tiene un caso así —la
factura de 80.995 unidades que la clase 4 decidió conservar—: basta que caiga del lado tapado.

**Tercera, y es la que importa para gestión: al decidir, los datos llegan de a uno.** En producción no
existe «todo el archivo». El sistema solo tiene los parámetros que aprendió en el pasado. Calcularlos
sobre el conjunto completo equivale a suponer que ya conocíamos la distribución de lo que todavía no
había pasado.

Y como el incumplimiento **no se nota en el resultado**, no sirve revisarlo después: hay que cumplirlo
por construcción. Eso es exactamente lo que hace el `Pipeline`.

Y se puede ver: **basta escribir el objeto** para que se dibuje. No hay que pedirlo con nada — el diagrama
es lo que scikit-learn muestra por defecto.
"""),
    code("""
Pipeline([('preparar', preprocesador(X_train)), ('modelo', LinearRegression())])
"""),
    md("""
# 4 · Ahora ustedes

Construyan el **precio medio de la factura** de la manera directa: el monto dividido por las unidades.
Incorpórenlo al flujo y midan.

<!-- contrato: accion=variable-de-la-actividad -->
"""),
    code("""
precio_efectivo = facturas["monto_total"] / facturas["unidades"]

X_train_efectivo = X_train.assign(precio_efectivo=precio_efectivo)
X_val_efectivo = X_val.assign(precio_efectivo=precio_efectivo)
"""),
    code("""
flujo_efectivo = ajustar(X_train_efectivo, y_train)

error_train_efectivo = error(flujo_efectivo, X_train_efectivo, y_train)
error_val_efectivo = error(flujo_efectivo, X_val_efectivo, y_val)
"""),
    code("""
display(Markdown(f'Entrenamiento **{num(error_train_efectivo, 1)}** · '
                 f'validación **{num(error_val_efectivo, 1)}** libras.'))
"""),
    md("""
La cifra se desploma. **La tarea no es celebrarlo: es decidir si es creíble.**

Y la respuesta no se busca en el error. Se busca en la **procedencia** de la columna. Comprueben esto:
"""),
    code("""
identidad = np.isclose(precio_efectivo * facturas["unidades"], facturas["monto_total"])
display(Markdown(f'`precio_efectivo × unidades = monto_total` se cumple en **{num(identidad.sum())} de '
                 f'{num(len(facturas))}** facturas, o sea el **{num(100 * identidad.mean())} %**.'))
"""),
    md("""
El 100 %, y no por casualidad: es una **identidad algebraica**. La columna contiene el objetivo. El
modelo no aprendió nada sobre el negocio; le entregamos la respuesta dividida por un número que también
le dimos.

Y noten el contraste con el precio de catálogo, que sí es legítimo: ese resume el histórico de **otras**
líneas, no la factura que se está midiendo.
"""),
    md("""
## Ahora la parte que se mira

La cifra bajó muchísimo, y eso **no** es lo que se les pide entregar.

Escriban, en una o dos frases:

- si la cifra es **creíble** o no, y por qué;
- **cuál de las formas del catálogo** de la clase es esta columna;
- **de dónde sale** el valor de cada una de las dos variables que probaron —la del catálogo y esta—, y
  en qué se diferencian.

Sin mencionar cuánto bajó el error.
"""),
    code("""
# escriban su diagnóstico acá, entre las comillas
diagnostico = \"\"\"

\"\"\"
print(diagnostico.strip() or 'Todavía sin escribir.')
"""),
    code("""
en_validacion = X_val_catalogo["precio_catalogo_medio"].notna()
sub = X_val_catalogo[en_validacion]
coincide = np.isclose(sub["precio_catalogo_medio"] * sub["unidades"], y_val[en_validacion])
lineas_validacion = lineas[lineas["n_documento"].isin(set(facturas["n_documento"].loc[X_val.index]))]
"""),
    code("""
display(Markdown(
    f'Líneas de validación que alimentan el catálogo aprendido: '
    f'**{lineas_validacion["n_documento"].isin(documentos_entrenamiento).sum()}**.\\n\\n'
    f'La identidad se cumple en {num(coincide.sum())} de {num(len(sub))} facturas de validación '
    f'(**{num(100 * coincide.mean(), 1)} %**), y esas son facturas de una línea de un producto de precio '
    f'estable: la mediana coincide con el precio. No es la factura mirándose a sí misma.'))
"""),
    md("""
Cero líneas de validación entran en el catálogo con el que se las mide. Y ese cero es **por
construcción**: el catálogo se arma con las líneas de las facturas de entrenamiento, así que no podía ser
otro. Lo que importa no es la sorpresa, es que es exactamente lo que la clase 5 no podía decir — allá el
catálogo se calculaba sobre todo el archivo.

# 5 · La prueba, una sola vez

Descartada la variable circular, la representación elegida es la del **precio de catálogo**. Recién ahora
abrimos el conjunto que estuvo tapado todo el rato.

<!-- contrato: accion=abrir-la-prueba -->
"""),
    code("""
error_prueba = error(flujo_catalogo, X_test_catalogo, y_test)      # la única vez que se toca la prueba
final = pd.DataFrame({'error mediano (libras)': [error_train_catalogo, error_val_catalogo, error_prueba]},
                     index=['entrenamiento', 'validación', 'prueba'])
"""),
    code("""
final.style.format(con_coma)
"""),
    code("""
display(Markdown(
    f'Prueba y validación se separan **{num(abs(error_prueba - error_val_catalogo), 1)} libras**, '
    f'y cambiar la semilla movía la cifra hasta {num(amplitud, 1)}.'))
"""),
    md("""
No son idénticas, y no tenían por qué serlo. Lo que importa es que se separan **menos** de lo que
mueve un cambio de semilla, así que la estimación se sostiene.

Y esa cifra ya no se toca. Si ahora probáramos otra representación y volviéramos a mirar la prueba, la
habríamos convertido en un segundo conjunto de validación.

## Para llevarse

- Medir donde se ajustó **describe**; medir del otro lado del muro **estima**.
- Todo lo que aprende un parámetro se ajusta con entrenamiento, y el `Pipeline` lo hace por construcción.
- El flujo previene la fuga de **procedimiento**. La de **origen** la atraviesa sin queja.
- La cifra no denuncia. Lo que permite confiar en ella es saber **de dónde viene cada variable**.

## Atribución de datos

- **Creador:** Chen, D. (2012)
- **Fuente:** [UCI Machine Learning Repository — *Online Retail II*, dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** obra derivada en dos pasos. Las ocho columnas originales se renombraron al español
  en `snake_case`, sin traducir los valores; después se aplicaron las siete decisiones de limpieza de la
  clase 4, que eliminan 5.268 filas repetidas y agregan seis columnas.
"""),
]


def llamada(fuente: str, nombre: str) -> dict:
    """Cuenta las llamadas a `nombre` y su cantidad de argumentos, por AST.

    LA CANTIDAD DE ARGUMENTOS TAMBIÉN SE DERIVA, no se escribe a mano. Estaba fija en 4 y 3, y al
    cambiar la firma de `medir` el contrato quedó describiendo un notebook que ya no existía: el
    checker buscaba llamadas de 4 argumentos y no encontraba ninguna. Derivarlo cierra esa puerta.
    Falla cerrado si las llamadas no coinciden entre sí, porque entonces un único número no las
    describe.
    """
    # Las líneas de magia (%%capture) y de shell (!pip) no son Python: se sacan antes de parsear.
    limpio = "\n".join(l for l in fuente.splitlines() if not l.lstrip().startswith(("%", "!")))
    tree = ast.parse(limpio)
    firmas = [
        len(nodo.args) + len(nodo.keywords)
        for nodo in ast.walk(tree)
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name) and nodo.func.id == nombre
    ]
    assert firmas, f"no hay llamadas a {nombre}: el contrato no puede declararlas"
    distintas = set(firmas)
    assert len(distintas) == 1, (
        f"las llamadas a {nombre} tienen {sorted(distintas)} argumentos; un solo número no las describe")
    return {"funcion": nombre, "argumentos": firmas[0], "cantidad": len(firmas)}


def contrato(celdas: list[dict]) -> dict:
    """Deriva `metadata.curso_contrato` de la lista FINAL de celdas.

    Se calcula, no se escribe a mano: agregar, quitar o mover una celda lo actualiza solo. Es la regla
    de `curso_checks` en `CLAUDE.md`, y existe porque un contrato con conteos de una versión anterior es
    peor que no tener contrato: pasa el chequeo y describe otro notebook.
    """
    codigo = [c for c in celdas if c["cell_type"] == "code"]
    # Se unen con salto de línea: sin él la última línea de una celda se pegaba a la primera de
    # la siguiente y el AST no parseaba.
    fuente = "\n".join("".join(c["source"]) for c in codigo)
    lineas_por_celda = [
        len([l for l in "".join(c["source"]).splitlines()
             if l.strip() and not l.strip().startswith("#")])
        for c in codigo
    ]
    mediana = statistics.median(lineas_por_celda)
    return {
        "forma": {
            "markdown": sum(1 for c in celdas if c["cell_type"] == "markdown"),
            "codigo": len(codigo),
            "llamadas": [llamada(fuente, "ajustar"), llamada(fuente, "train_test_split")],
        },
        "acciones": [
            {"marcador": "fijar-base", "asignacion": "lineas"},
            {"marcador": "agregar-por-factura", "codigo": "groupby"},
            {"marcador": "partir-en-tres", "codigo": "train_test_split"},
            {"marcador": "armar-el-flujo", "codigo": "Pipeline"},
            {"marcador": "variable-de-la-actividad", "asignacion": "precio_efectivo"},
            {"marcador": "abrir-la-prueba", "asignacion": "final"},
        ],
        # Es una MEDIANA, no un máximo por celda: deja pasar el setup largo y sigue exigiendo que el
        # cuaderno esté partido en unidades chicas. Se fija al valor real medido, con un margen de 1.
        "max_mediana_lineas_codigo": int(mediana) + 1,
    }


def main() -> None:
    celdas = con_ids(CELDAS)
    nb = {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "curso_contrato": contrato(celdas),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    forma = nb["metadata"]["curso_contrato"]["forma"]
    print(f"escrito {SALIDA.relative_to(REPO_ROOT)}")
    print(f"  {len(celdas)} celdas · {forma['markdown']} markdown · {forma['codigo']} código")
    print(f"  contrato: mediana de líneas "
          f"{nb['metadata']['curso_contrato']['max_mediana_lineas_codigo'] - 1}"
          f" · llamadas {forma['llamadas']}")


if __name__ == "__main__":
    main()
