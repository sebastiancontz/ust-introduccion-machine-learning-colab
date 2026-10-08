#!/usr/bin/env python3
"""Construye notebooks/06-regresion-baseline.ipynb.

POR QUÉ EXISTE UN GENERADOR: el `.ipynb` no se edita a mano. Si hay que corregir algo, se corrige acá y
se vuelve a ejecutar. En la clase 1 se perdieron dos correcciones por parchear el artefacto.

ESTRUCTURA: los tres tiempos de la `practica` del temario, más el cierre. (1) El docente pone el
BASELINE primero, con las dos estrategias para que se vea que no da lo mismo, y recién después cambia
la última pieza del flujo por `LinearRegression`. (2) Entre todos se agregan las métricas CON NOMBRE
—MAE y RMSE— y entran Ridge y Lasso, cada uno con SU grilla de alpha. (3) Cada estudiante decide qué
modelo llevaría a una gerencia y lo defiende por escrito. Cierre: se abre la prueba UNA vez.

EL FLUJO DE LA SESIÓN 5 SE REUTILIZA, NO SE REEXPLICA: el `Pipeline` y el `ColumnTransformer` se dan
por sabidos. Entre modelos cambia la última pieza, más el escalado que exigen Ridge y Lasso, y el
notebook lo dice así porque es la idea que sostiene toda la comparación.

LA PARTICIÓN ES TRIPLE, 60 / 20 / 20, igual que en la clase 5. El alpha se elige sobre VALIDACIÓN y la
PRUEBA se abre una sola vez, al final. Elegir el alpha mirando prueba es DATA SNOOPING (Géron cap. 2
«Create a Test Set»; ISLP cap. 6 §6.1.3 p. 236) y deja el error falsamente optimista: medido en
este archivo, la mejora de Ridge pasa de 0,0044 honesta a 0,0064 espiando prueba.

DOS GRILLAS DE ALPHA, UNA POR MODELO, y no es un detalle: scikit-learn divide el RSS de Lasso por 2n
antes de sumar la penalización y el de Ridge no, así que el mismo número regulariza muy distinto en
cada uno. Con una grilla común, Ridge se ve inofensivo y Lasso destructivo, y las dos lecturas son de
la grilla y no del modelo.

LA CIFRA NUNCA SE ESCRIBE A MANO: todas se calculan en la celda y se muestran con Markdown dinámico
por f-string. Las del deck salen de `assets/ilustraciones/_build_c06_figuras.py`, que las imprime y las
protege con aserciones; acá se RECALCULAN, que es lo que pide la Fase 5D.

FRONTERAS, heredadas del `alcance_no` y verificadas al escribir:
  · No hay validación cruzada ni `GridSearchCV`: es la sesión 8.
  · No aparece R² como métrica de decisión.
  · No se enseña la familia de conteo que le correspondería al objetivo: se nombra y queda fuera.
  · No se interpreta ningún coeficiente como efecto causal.

Uso:
    conda run -n ust-ml python ediciones/2026/notebooks/_build_06_regresion_baseline.py
      (después: `make nb C=06` para validarlo y `make nb-check C=06` para el contrato)
"""
from __future__ import annotations

import ast
import base64
import json
import statistics
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
LOGO = "data:image/svg+xml;base64," + base64.b64encode(
    (REPO_ROOT / "assets" / "logo-ust.svg").read_bytes()).decode("ascii")  # embebido: Colab no muestra el SVG externo
SALIDA = AQUI / "06-regresion-baseline.ipynb"
ILUSTRACIONES = REPO_ROOT / "assets" / "ilustraciones"


def figura_base64(nombre: str, alt: str, width: int = 860) -> str:
    """Diagrama embebido como data URI: el notebook queda autocontenido y no depende de Pages.

    `alt` es obligatorio: sin él la figura no existe para quien usa lector de pantalla.
    """
    b64 = base64.b64encode((ILUSTRACIONES / nombre).read_bytes()).decode("ascii")
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
    md(f"""
<div style="display:flex; align-items:center; gap:18px; text-align:left">
<img src="{LOGO}" width="100" alt="Logo de la Universidad Santo Tomás">
<div>
<p>Ingeniería en Información y Control de Gestión</p>
<p>Facultad de Economía y Negocios</p>
<p>Introducción a Machine Learning</p>
<p>Semana 06: Regresión predictiva — baseline, métricas y regularización</p>
</div>
</div>
"""),
    md("""
# 06 · Regresión predictiva: baseline, métricas y regularización

El flujo de la clase anterior no cambia. Entre un modelo y otro cambia **la última pieza** del
`Pipeline`, el modelo, y Ridge y Lasso suman el escalado que exigen. Con eso alcanza para recorrer
cuatro modelos.
"""),
    code("""
%%capture
!pip install -q pandas numpy pyarrow scikit-learn
"""),
    code("""
from IPython.display import Markdown, display  # muestra frases con las cifras ya calculadas

def num(x, dec=0):
    \"\"\"miles con punto, decimales con coma y el menos tipográfico, como el resto del material\"\"\"
    texto = f'{x:,.{dec}f}'.replace(',', '@').replace('.', ',').replace('@', '.')
    return texto.replace('-', '\u2212')

def dos(x):
    \"\"\"la versión de dos decimales que usan las tablas\"\"\"
    return num(x, 2)
""",
         ),
    md("""
## El encuadre, declarado antes de tocar el archivo

Sin esto no se puede leer ninguna cifra de la actividad, así que va primero:

- La **unidad de observación es el cliente**. Un registro es un cliente, no una factura ni una línea
  de factura.
- El **objetivo es un conteo**: cuántas veces vuelve a comprar cada cliente en la ventana futura.
- El **momento de la predicción** es el corte entre las dos ventanas. Las fuentes permitidas son solo
  lo anterior a él.
- Por eso la **población no se filtra por lo que pasó después**. El cliente que no vuelve **vale
  cero**, no desaparece del archivo. Filtrarlo sería seleccionar por el resultado, que es justo la
  falta que la clase anterior enseñó a detectar.
"""),
    md("""
### Cargar el archivo

Lo leemos con **pandas**, la librería de tablas del curso. Es un archivo por cliente, construido y
versionado, con el corte temporal ya aplicado. Tres de sus
columnas son un RFM —`recencia_dias`, `facturas_base` y `monto_base`—: se crearon agregando por
cliente las compras anteriores al corte, que es ingeniería de variables.

<!-- contrato: accion=fijar-base -->
"""),
    code("""
import os                # revisa si el archivo está en una carpeta local
import pandas as pd      # la librería de tablas: lee el archivo y lo deja como DataFrame

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

clientes = pd.read_parquet(BASE + 'clientes_ventas.parquet')
print('Filas y columnas:', clientes.shape)
clientes.head()
"""),
    code("""
ceros = int((clientes['facturas_futuras'] == 0).sum())
display(Markdown(
    f'De los **{num(len(clientes))}** clientes del archivo, **{num(ceros)}** no vuelven a comprar en '
    f'la ventana futura: el **{num(100 * ceros / len(clientes))} %**. Son el valor más frecuente '
    f'del objetivo y no se pueden sacar.'
))
"""),
    md("""
### La partición, igual que la clase anterior

De aquí en adelante casi todo viene de **scikit-learn** (`sklearn`), la librería de machine
learning del curso. Cada pieza se importa en la celda donde se usa por primera vez.

Tres pedazos y cada uno con su trabajo. **Entrenamiento** ajusta, **validación** decide entre
alternativas, y **prueba** se abre **una sola vez** cuando ya no queda nada por decidir.

Que validación exista es lo que nos va a permitir elegir un parámetro sin hacer trampa. Si lo
eligiéramos mirando prueba, el error que reportamos dejaría de estimar lo que va a pasar con clientes
nuevos: quedaría optimista por construcción.

<!-- contrato: accion=partir-en-tres -->
"""),
    code("""
from sklearn.model_selection import train_test_split  # sortea la partición

COLUMNAS_NUM = ['facturas_base', 'monto_base', 'productos_base',
                'ticket_medio_base', 'recencia_dias', 'antiguedad_dias']
COLUMNAS_CAT = ['mercado']

X = clientes[COLUMNAS_NUM + COLUMNAS_CAT]
y = clientes['facturas_futuras']

# 20 % a prueba, y del resto un 25 % a validación: quedan 60 / 20 / 20
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

print('entrenamiento:', len(X_train), '· validación:', len(X_val), '· prueba:', len(X_test))
"""),
    md(figura_base64(
        "c06-particion.svg",
        "Tres filas de barras horizontales. La primera muestra el archivo completo con 3.314 "
        "clientes. La segunda lo parte en dos: entrenamiento más validación con 2.651 a la "
        "izquierda y prueba con 663 a la derecha. La tercera vuelve a partir solo el pedazo "
        "izquierdo, quedando entrenamiento con 1.988, validación con 663 y prueba intacta con 663. "
        "Debajo, cada pedazo dice para qué sirve: entrenamiento ajusta el modelo, validación decide "
        "entre alternativas y prueba se abre una sola vez.")
       + """

Las dos líneas de arriba son **dos particiones, no una**. La segunda cae solo sobre el pedazo
izquierdo de la primera, y por eso `test_size=0.25` sobre el 80 % que quedaba deja un 20 % del total.

Prueba no se toca en ninguno de los dos pasos siguientes. Ésa es toda la disciplina.
"""),
    md("""
# Tiempo 1 · Yo lo hago

## El baseline primero, y con la estrategia que corresponde

La clase anterior ya usó un predictor constante para leer sus cifras contra algo. Hoy ese baseline
deja de ser un dato al costado: es el **competidor**.

El flujo es el mismo de la sesión 5: un `Pipeline` con dos pasos con nombre, **`preparar`** y
**`modelo`**. La preparación se escribe una vez acá y la comparten los modelos que no necesitan
escalar. Ridge y Lasso, más abajo, van a tener la suya, con `StandardScaler`.

Y antes de entrenar hay que **declarar la métrica**, no después. Nuestra vara es el **promedio del
error absoluto, en facturas por cliente**.

<!-- contrato: accion=armar-el-flujo -->
"""),
    code("""
from sklearn.compose import ColumnTransformer  # prepara cada grupo de columnas a su manera
from sklearn.preprocessing import OneHotEncoder  # convierte una categórica en columnas de 0 y 1

# las numéricas pasan tal cual; mercado entra en k−1 columnas, con exportación de referencia
preparacion = ColumnTransformer([
    ('num', 'passthrough', COLUMNAS_NUM),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUMNAS_CAT),
])
"""),
    md("""
La métrica declarada tiene una consecuencia que casi nunca se dice: **para un predictor constante, el
mejor depende de la vara**. El promedio minimiza el error al cuadrado y la mediana minimiza el
error absoluto. Como medimos con la absoluta, el baseline que corresponde es la **mediana**.

Vamos a calcular los dos para que se vea que no da lo mismo. El error se mide con
`mean_absolute_error`, que calcula justamente el promedio del error absoluto.

<!-- contrato: accion=el-baseline -->
"""),
    code("""
from sklearn.pipeline import Pipeline            # encadena la preparación y el modelo
from sklearn.dummy import DummyRegressor         # el modelo que predice siempre lo mismo
from sklearn.metrics import mean_absolute_error  # el promedio del error absoluto

piso_promedio = Pipeline([
    ('preparar', preparacion),
    ('modelo', DummyRegressor(strategy='mean')),
])
piso_mediana = Pipeline([
    ('preparar', preparacion),
    ('modelo', DummyRegressor(strategy='median')),
])
piso_promedio.fit(X_train, y_train)
piso_mediana.fit(X_train, y_train)

# mean_absolute_error recibe primero lo observado y después lo predicho, en ese orden
mae_promedio = mean_absolute_error(y_val, piso_promedio.predict(X_val))
mae_mediana = mean_absolute_error(y_val, piso_mediana.predict(X_val))
print('predecir siempre el promedio:', dos(mae_promedio))
print('predecir siempre la mediana: ', dos(mae_mediana))
"""),
    md("""
## Ahora cambiamos la última pieza

Elegir el baseline del promedio nos daría un rival debilitado, y todo lo que el modelo le ganara de
más sería mérito prestado. Seguimos con la **mediana**.

El flujo es el mismo. En el paso `modelo`, `DummyRegressor` sale y entra `LinearRegression`, que
es el modelo que vieron en Estadística y Econometría, ahora con otro propósito: **predecir**, no explicar.

<!-- contrato: accion=el-modelo -->
"""),
    code("""
from sklearn.linear_model import LinearRegression  # la regresión lineal, ajustada por OLS

lineal = Pipeline([
    ('preparar', preparacion),
    ('modelo', LinearRegression()),    # la única pieza que cambia
])
lineal.fit(X_train, y_train)
mae_lineal = mean_absolute_error(y_val, lineal.predict(X_val))

display(Markdown(
    f'El baseline de la mediana estaba en **{dos(mae_mediana)}** facturas de error. El modelo lineal '
    f'baja a **{dos(mae_lineal)}**: le gana por **{dos(mae_mediana - mae_lineal)}**. Recién ahora esa '
    f'cifra de error significa algo, porque tiene contra qué leerse.'
))
"""),
    md("""
### Leer los coeficientes, con una cautela

`coef_` guarda un número por columna del modelo ya ajustado, y `get_feature_names_out` nos dice a
qué columna corresponde cada uno. Los juntamos para poder leerlos, y los ordenamos con `sort_values`
por **tamaño**, sin importar el signo.

**Ojo con ese orden:** el tamaño de un coeficiente depende de la **unidad** de su columna. Uno por libra
se ve chico aunque mueva mucho la predicción, porque los montos son números grandes. Así que el orden
no dice qué columna pesa más: para comparar hay que traducir cada coeficiente a un cambio con sentido,
como hacen las slides.

**La cautela:** un coeficiente describe una **asociación condicional**, no un efecto. Decir «con todo
lo demás igual» está bien; decir «si subimos esta variable, el objetivo sube» no lo está, porque
nadie intervino nada.
"""),
    code("""
nombres = lineal.named_steps['preparar'].get_feature_names_out()
coeficientes = pd.DataFrame({'coeficiente': lineal.named_steps['modelo'].coef_}, index=nombres)
# ordenados por tamaño, sin importar el signo; el tamaño depende de la unidad de cada columna
coeficientes = coeficientes.sort_values('coeficiente', key=abs, ascending=False)
coeficientes
"""),
    code("""
principal = coeficientes.index[0].split('__')[-1]     # sin el prefijo que agrega la preparación
valor = coeficientes['coeficiente'].iloc[0]
display(Markdown(
    f'El coeficiente más grande en magnitud es el de **{principal}**, con **{num(valor, 3)}**. Se lee '
    f'así: entre dos clientes que coinciden en todo lo demás, el que tiene una unidad más de esa '
    f'variable se asocia, en promedio, a {num(valor, 3)} facturas más en la ventana futura. Es una '
    f'comparación entre clientes, no el resultado de intervenir sobre uno.'
))
"""),
    md("""
# Tiempo 2 · Lo hacemos juntos

## Dos métricas en la misma tabla

Hasta acá usamos una sola vara. Ahora le ponemos nombre a las dos que vamos a usar:

- **MAE** (*mean absolute error*), el **promedio del error absoluto**. Es la que veníamos usando.
- **RMSE** (*root mean squared error*), la **raíz del error cuadrático medio**. Eleva cada error al
  cuadrado antes de promediar, así que un error grande pesa mucho más que varios chicos.

`mean_absolute_error` y `root_mean_squared_error` las calculan, y las dos reciben lo mismo y en el
mismo orden: primero lo **observado** y después lo **predicho**. Hasta ahora solo usamos la primera.

<!-- contrato: accion=metricas-con-nombre -->
"""),
    code("""
from sklearn.metrics import root_mean_squared_error  # la raíz del error cuadrático medio

dos_modelos = {'baseline (mediana)': piso_mediana, 'regresión lineal': lineal}

filas = []
for nombre, modelo in dos_modelos.items():
    prediccion = modelo.predict(X_val)
    filas.append({'modelo': nombre,
                  'MAE': mean_absolute_error(y_val, prediccion),
                  'RMSE': root_mean_squared_error(y_val, prediccion)})

tabla = pd.DataFrame(filas)
tabla.round(2)
"""),
    md("""
**Para discutir:** en las dos filas el RMSE es más alto que el MAE. ¿Por qué? ¿Y la brecha entre las
dos se abre más en el baseline o en el modelo? Pista: el RMSE eleva cada error al cuadrado, y nuestro
objetivo tiene cola larga.
"""),
    md("""
## Ponerle precio a la complejidad

Ridge y Lasso son el mismo modelo lineal con una **penalización** añadida: además de acercarse a los datos,
tienen que mantener chicos sus coeficientes.

- `Ridge` penaliza los coeficientes **al cuadrado**. Los encoge a todos y ninguno sale del modelo.
- `Lasso` penaliza su **valor absoluto**. Encoge y además lleva algunos exactamente a cero, con lo
  cual saca variables.

Las dos **exigen escalar** las variables, y por eso llevan su propia preparación: la misma de
arriba, con `StandardScaler` en las numéricas, que las deja todas en la misma escala. Sin eso, la
penalización castigaría a la variable que quedó con números grandes por su unidad, no por su
importancia.
"""),
    code("""
from sklearn.preprocessing import StandardScaler  # deja cada numérica con promedio 0 y desviación 1

# la misma preparación, con las numéricas estandarizadas
preparacion_escalada = ColumnTransformer([
    ('num', StandardScaler(), COLUMNAS_NUM),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUMNAS_CAT),
])
"""),
    md("""
### Y el alpha se elige sobre validación

> **Antes de correr esto, un aviso.** Este cuaderno parte el archivo en 60/20/20 de una vez, y las
> slides parten 75/25 y después recortan la validación del entrenamiento. Como la partición es otra,
> **el alpha que gane acá puede no ser el de las slides**. Eso no es un error: es exactamente lo que
> dice el cierre de la clase, que una partición al azar tiene su propia suerte. Cuando
> vean la diferencia, esa es la lección, no una contradicción.

`alpha` es el precio de la penalización. Dos cosas antes de moverlo:

1. **Cada modelo tiene su propia escala de alpha.** No son comparables entre sí, así que cada uno va
   con su grilla. Buscarlos en una grilla común es la forma más rápida de concluir algo falso.
2. **La elección se hace sobre validación**, con prueba todavía cerrada. Elegir el que da el mejor
   error de prueba y después reportar ese mismo error es hacerse trampa a uno mismo.

<!-- contrato: accion=elegir-alpha -->
"""),
    code("""
from sklearn.linear_model import Ridge  # regresión lineal que penaliza los coeficientes al cuadrado

ALPHAS_RIDGE = [10.0, 100.0, 1000.0]

filas_ridge = []
alpha_ridge = None
mejor_error = None
for a in ALPHAS_RIDGE:
    candidato = Pipeline([
        ('preparar', preparacion_escalada),
        ('modelo', Ridge(alpha=a)),
    ])
    candidato.fit(X_train, y_train)
    error = mean_absolute_error(y_val, candidato.predict(X_val))
    filas_ridge.append({'alpha': a, 'MAE en validación': error})
    # nos quedamos con el alpha del error más chico
    if mejor_error is None or error < mejor_error:
        alpha_ridge = a
        mejor_error = error

print('alpha elegido para Ridge:', alpha_ridge)
"""),
    code("""
from sklearn.linear_model import Lasso  # regresión lineal que penaliza su valor absoluto

# el mismo recorrido para Lasso, con su propia grilla
ALPHAS_LASSO = [0.01, 0.1, 1.0]

filas_lasso = []
alpha_lasso = None
mejor_error = None
for a in ALPHAS_LASSO:
    candidato = Pipeline([
        ('preparar', preparacion_escalada),
        ('modelo', Lasso(alpha=a)),
    ])
    candidato.fit(X_train, y_train)
    error = mean_absolute_error(y_val, candidato.predict(X_val))
    filas_lasso.append({'alpha': a, 'MAE en validación': error})
    if mejor_error is None or error < mejor_error:
        alpha_lasso = a
        mejor_error = error

print('alpha elegido para Lasso:', alpha_lasso)
"""),
    code("""
todas = []
for nombre, filas in [('Ridge', filas_ridge), ('Lasso', filas_lasso)]:
    for fila in filas:
        todas.append({'modelo': nombre,
                      'alpha': fila['alpha'],
                      'MAE en validación': fila['MAE en validación']})

pd.DataFrame(todas).round(4)
"""),
    code("""
ridge = Pipeline([
    ('preparar', preparacion_escalada),
    ('modelo', Ridge(alpha=alpha_ridge)),
])
lasso = Pipeline([
    ('preparar', preparacion_escalada),
    ('modelo', Lasso(alpha=alpha_lasso)),
])
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

coef_lasso = pd.DataFrame({'coeficiente': lasso.named_steps['modelo'].coef_}, index=nombres)
en_cero = list(coef_lasso[coef_lasso['coeficiente'].abs() < 1e-10].index)
# no lo afirmamos: lo contamos, igual que el de Lasso
vivos_ridge = int((abs(ridge.named_steps['modelo'].coef_) > 1e-10).sum())
display(Markdown(
    f'Lasso con alpha {alpha_lasso} llevó a cero **{len(en_cero)} de {len(coef_lasso)}** coeficientes: '
    f'{", ".join(en_cero) if en_cero else "ninguno"}. Ridge, en cambio, conserva '
    f'{vivos_ridge} de {len(coef_lasso)}: encogió, pero no sacó a nadie.'
))
"""),
    md("""
# Tiempo 3 · Lo hacen ustedes

## La decisión

Tienen cuatro modelos sobre la mesa y hay que **llevar uno a una gerencia**. La actividad es
decidirlo y defenderlo por escrito en **tres líneas**.

La decisión **no es obvia**, y ahí está el punto. Miren la tabla de alpha que calcularon más arriba
antes de responder:

- El MAE más bajo es **un** argumento, y no siempre gana por mucho.
- Un modelo que usa **menos columnas** es más fácil de explicar y de mantener. Ése es un argumento
  distinto —**simplicidad**— y hay que decidir cuánto pesa cuando la diferencia de error es chica.
- Y ojo con esto: **el alpha lo eligió una sola partición de validación**. Con otra partición al azar el
  ganador puede ser otro. Cuánta confianza le dan a esa elección es parte de la decisión: si gana el
  modelo más simple de todos, pregúntense si le creerían igual con clientes nuevos. El cierre lo pone a
  prueba.

**Criterio de éxito:** elegir el baseline que corresponde a la métrica y justificarlo; leer un
coeficiente en unidades del negocio, con su cautela; leer MAE y RMSE y explicar por qué difieren;
nombrar qué le hace la penalización a los coeficientes y en qué se diferencian Ridge y Lasso; y
defender una elección de modelo con la tabla delante.

**Evidencia observable:** la tabla comparativa de los cuatro modelos con sus dos métricas, la lista
de coeficientes que Lasso llevó a cero, una frase que lea un coeficiente, y las tres líneas de la
decisión justificada.

**Una advertencia antes de mirar la tabla.** Las filas de Ridge y Lasso salen un poco optimistas,
porque su alpha se eligió con esta misma validación: la tabla sirve para **decidir**, no para reportar.
La cifra que se reporta llega al final, con la prueba.
"""),
    code("""
# armen la tabla de los cuatro modelos sobre validación y mírenla antes de decidir
cuatro_modelos = {'baseline (mediana)': piso_mediana,
                  'regresión lineal': lineal,
                  f'ridge (alpha={alpha_ridge})': ridge,
                  f'lasso (alpha={alpha_lasso})': lasso}

filas = []
for nombre, modelo in cuatro_modelos.items():
    prediccion = modelo.predict(X_val)
    filas.append({'modelo': nombre,
                  'MAE': mean_absolute_error(y_val, prediccion),
                  'RMSE': root_mean_squared_error(y_val, prediccion)})

comparacion = pd.DataFrame(filas)
# el baseline no usa ningún predictor; los otros, los que no quedaron en cero
comparacion['predictores'] = [0, len(nombres), len(nombres), len(nombres) - len(en_cero)]
comparacion.round(2)
"""),
    md("""
### Su respuesta

Escriban acá las tres líneas. Qué modelo llevarían, con qué cifra lo sostienen, y qué renuncian al
elegirlo.

1.
2.
3.

Y una frase más: lean **un coeficiente** del modelo que eligieron, con su unidad, con «con todo lo
demás igual», y diciendo por qué no es el efecto de intervenir.

-
"""),
    md("""
# Cierre

## Abrir la prueba, una sola vez

Ya no queda nada por decidir: la métrica está declarada, el baseline elegido y los alpha fijados
sobre validación.

Falta **un paso** antes de medir, y es el que más se olvida. Validación ya hizo su trabajo: eligió el
alpha. Así que ahora se **reajustan los cuatro modelos con todo el entrenamiento** —entrenamiento más
validación— y recién esos son los que se llevan a prueba. Reservar los clientes de validación, que ya no
hacen falta, sería tirar datos.

**Una precaución de scikit-learn.** Un `Pipeline` ajusta el mismo objeto de preparación que recibe:
no hace una copia. Si estos modelos finales usaran `preparacion_escalada`, ajustarlos con todo el
entrenamiento cambiaría también, sin aviso, a los `ridge` y `lasso` de arriba. Por eso las dos
preparaciones se escriben de nuevo.
"""),
    code("""
# preparaciones nuevas, para no tocar las que ya ajustaron los modelos de arriba
preparacion_final = ColumnTransformer([
    ('num', 'passthrough', COLUMNAS_NUM),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUMNAS_CAT),
])
preparacion_escalada_final = ColumnTransformer([
    ('num', StandardScaler(), COLUMNAS_NUM),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUMNAS_CAT),
])
"""),
    code("""
# paso 4 · con los alpha ya elegidos, se reajusta con todo el entrenamiento
finales = {
    'baseline (mediana)': Pipeline([('preparar', preparacion_final),
                                    ('modelo', DummyRegressor(strategy='median'))]),
    'regresión lineal': Pipeline([('preparar', preparacion_final),
                                  ('modelo', LinearRegression())]),
    f'ridge (alpha={alpha_ridge})': Pipeline([('preparar', preparacion_escalada_final),
                                              ('modelo', Ridge(alpha=alpha_ridge))]),
    f'lasso (alpha={alpha_lasso})': Pipeline([('preparar', preparacion_escalada_final),
                                              ('modelo', Lasso(alpha=alpha_lasso))]),
}

for nombre, modelo in finales.items():
    finales[nombre] = modelo.fit(X_trainval, y_trainval)

print('reajustados con', len(X_trainval), 'clientes en vez de', len(X_train))
"""),
    md("""
Recién ahora se abre el conjunto de prueba, y se abre una vez.

<!-- contrato: accion=abrir-la-prueba -->
"""),
    code("""
# el mismo bucle de recién: cambian el conjunto y los modelos, que ahora son los reajustados
filas = []
for nombre, modelo in finales.items():
    prediccion = modelo.predict(X_test)
    filas.append({'modelo': nombre,
                  'MAE': mean_absolute_error(y_test, prediccion),
                  'RMSE': root_mean_squared_error(y_test, prediccion)})

prueba = pd.DataFrame(filas)
prueba.round(2)
"""),
    md("""
## Un límite del modelo, a la vista

Mirar dónde se equivoca el modelo sirve para nombrar su límite, no para arreglarlo hoy. El más visible
está en sus predicciones de prueba.
"""),
    code("""
prediccion = finales['regresión lineal'].predict(X_test)
negativas = int((prediccion < 0).sum())
aviso = (f'llega a predecir **{dos(prediccion.min())}** facturas, y nadie compra un número negativo de '
         f'veces' if negativas else
         f'no llegó a predecir ningún negativo en esta partición, pero su mínimo es '
         f'**{dos(prediccion.min())}** y nada en el modelo lo impide')
display(Markdown(
    f'De los **{num(len(X_test))}** clientes de prueba, el modelo lineal predice '
    f'**{num(negativas)}** valores negativos: {aviso}. Es el límite de haber tratado un **conteo** '
    f'como si fuera una cantidad continua. La familia de modelos que le corresponde existe y queda '
    f'fuera de esta clase.'
))
"""),
    md("""
## La pregunta que queda abierta

Los alpha de hoy salieron de **una** partición de validación, y esa partición tiene su propia
suerte: con otra, el elegido podría ser otro. ¿Cómo se elige un parámetro sin depender de una sola
partición?

Eso es la sesión 8, con validación cruzada y búsqueda sobre grilla.
"""),
    md("""
## Atribución de datos

- **Creador:** Chen, D. (2012)
- **Fuente:** [UCI Machine Learning Repository — *Online Retail II*, dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** obra derivada en tres pasos. Las columnas originales se renombraron al español en
  `snake_case`; se aplicaron las siete decisiones de limpieza de la clase 4; y
  `_prep_clientes_ventas.py` agregó por cliente con un corte temporal, quedando 3.314 registros y
  9 columnas.
"""),
]


def llamada(fuente: str, nombre: str) -> dict:
    """Cuenta las llamadas a `nombre` y su cantidad de argumentos, por AST.

    LA CANTIDAD DE ARGUMENTOS TAMBIÉN SE DERIVA, no se escribe a mano: un contrato con la firma de
    una versión anterior describe otro notebook y pasa el chequeo igual. Falla cerrado si las
    llamadas no coinciden entre sí, porque entonces un único número no las describe.
    """
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

    Se calcula, no se escribe a mano: agregar, quitar o mover una celda lo actualiza solo.
    """
    codigo = [c for c in celdas if c["cell_type"] == "code"]
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
            "llamadas": [llamada(fuente, "mean_absolute_error"), llamada(fuente, "train_test_split")],
        },
        "acciones": [
            {"marcador": "fijar-base", "asignacion": "clientes"},
            {"marcador": "partir-en-tres", "codigo": "train_test_split"},
            {"marcador": "armar-el-flujo", "codigo": "ColumnTransformer"},
            {"marcador": "el-baseline", "codigo": "DummyRegressor"},
            {"marcador": "el-modelo", "codigo": "LinearRegression"},
            {"marcador": "metricas-con-nombre", "codigo": "root_mean_squared_error"},
            {"marcador": "elegir-alpha", "asignacion": "alpha_ridge"},
            {"marcador": "abrir-la-prueba", "asignacion": "prueba"},
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
