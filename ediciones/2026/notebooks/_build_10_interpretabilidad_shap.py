#!/usr/bin/env python3
"""Construye notebooks/10-interpretabilidad-shap.ipynb.

POR QUÉ EXISTE UN GENERADOR: el `.ipynb` no se edita a mano. Si hay que corregir algo, se corrige acá y
se vuelve a ejecutar con `make nb-build C=10`.

ESTRUCTURA: los tres tiempos de la `practica` del temario, más el cierre. (1) El docente reentrena el
bosque ganador de la sesión 8, muestra en una celda que la importancia nativa no tiene signo ni caso,
instancia `TreeExplainer`, comprueba la aditividad y lee el enjambre. (2) Entre todos: la tabla de
descomposición y la cascada de dos clientes contrastantes, y el gráfico de dependencia de `meses_mora`
con su punto de quiebre calculado. (3) La actividad: los 10 peores errores de cada dirección, ya
filtrados y ordenados (andamiaje de la `calibracion`), sus cascadas y la tabla diagnóstica.

LA MEDICIÓN ES LA DEL GENERADOR DE FIGURAS (`assets/ilustraciones/_build_c10_figuras.py`): mismo
archivo, misma partición, mismo bosque y la explicación sobre `X_test`. Por eso las cifras del cuaderno
coinciden con las del deck sin copiarse: se RECALCULAN en cada celda y se muestran con Markdown dinámico
por f-string.

PRUEBA SE LEE VARIAS VECES, a propósito: el modelo está congelado desde la sesión 8 y la `calibracion`
encuadra la explicación como auditoría final de aceptación sobre `X_test`. Nada se ajusta con lo que se
ve; el cuaderno lo dice en el encuadre. Por eso el contrato no declara `max_celdas_lectura`.

LA PREPARACIÓN SALE COMO TABLA (`set_output(transform='pandas')` y `sparse_output=False`): así SHAP
recibe los nombres de las columnas y los gráficos los muestran. No cambia el ajuste del bosque.

PYTHON BÁSICO (docente, 2026-09-09): sin listas por comprensión ni `lambda`.

FRONTERAS, heredadas del `alcance_no` y verificadas al escribir:
  · Ni permutation importance ni ICE implementados; ni LIME; ni la teoría de los valores de Shapley.
  · La importancia nativa aparece una vez, para mostrar lo que no dice (lo pide la `practica`).
  · El gráfico de dependencia se lee como asociación observada, nunca como efecto de mover la variable.

Uso:
    make nb-build C=10   (regenera, ejecuta en sandbox y deja el ejecutado publicado)
    make nb-check C=10   (verifica el contrato)
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
SALIDA = AQUI / "10-interpretabilidad-shap.ipynb"


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
<p>Semana 10: Interpretabilidad con SHAP</p>
</div>
</div>
"""),
    md("""
# 10 · Interpretabilidad con SHAP

> **La pregunta de hoy:** Cobranza ya usa el random forest para decidir a quién llamar primero. Si
> pregunta por qué marcó a un cliente con 85 %, ¿qué le podemos responder y qué no podemos prometerle?

Tres tiempos: **yo lo hago** (abrir el modelo y leer qué usa en conjunto), **lo hacemos juntos**
(explicar dos clientes y leer la relación con la mora) y **lo hacen ustedes** (auditar los peores
errores y escribir qué falta en el archivo).
"""),
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas; la celda queda para correr fuera de Colab. La siguiente
define `num`, que escribe las cifras con punto de miles y coma decimal, y `pts`, que escribe una
diferencia de probabilidad en **puntos porcentuales** con su signo.
"""),
    code("""
%%capture
!pip install -q pandas numpy scikit-learn matplotlib shap
"""),
    code("""
from IPython.display import Markdown, display  # muestra frases con las cifras ya calculadas

def num(x, dec=0):
    # miles con punto, decimales con coma y el menos tipográfico
    texto = f'{x:,.{dec}f}'.replace(',', '@').replace('.', ',').replace('@', '.')
    return texto.replace('-', '−')

def pts(x):
    # 0,266 de probabilidad se escribe +26,6 puntos
    return ('+' if x >= 0 else '') + num(100 * x, 1)
"""),
    md("""
## Cargar datos

La cartera de cobranza de **tarjetas de crédito**, `riesgo_impago.csv`, la misma de las sesiones 7 y 8.

<!-- contrato: accion=fijar-base -->
"""),
    code("""
import os                # revisa si el archivo está en una carpeta local
import pandas as pd      # la librería de tablas

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

datos = pd.read_csv(BASE + 'riesgo_impago.csv')
print('registros y columnas:', datos.shape)
"""),
    md("""
## El encuadre: lo que se conserva y lo que empieza

- **Unidad de observación:** un registro es un **cliente observado en un mes**.
- **Objetivo:** si el cliente **incumple el pago del mes siguiente** (`incumplio_pago`). La clase
  positiva es **incumplir**: todo lo que explicamos hoy es la **probabilidad de impago**.
- **Se conserva:** el archivo, la partición con semilla 42 y el **bosque que ganó la sesión 8**. Hoy no
  se compite ni se ajusta nada.
- **Empieza:** abrir ese modelo y explicar qué usa, para poder defenderlo.

<!-- contrato: accion=declarar-positiva -->
"""),
    code("""
y = datos['incumplio_pago'].map({'no': 0, 'si': 1})  # 1 = incumple, la clase positiva

COLUMNAS_CAT = ['nivel_educacion']
COLUMNAS_NUM = []
for columna in datos.columns:  # todas menos el identificador, el objetivo y la categórica
    if columna not in ['id_cliente', 'incumplio_pago'] + COLUMNAS_CAT:
        COLUMNAS_NUM.append(columna)
X = datos[COLUMNAS_NUM + COLUMNAS_CAT]
"""),
    md("""
### La partición de siempre

Es la de las sesiones 7 y 8: 20 % a prueba, estratificada y con `random_state=42`. Hoy basta con el primer
corte: `X_trainval` (entrenamiento y validación juntos), con el que la sesión 8 reentrenó al ganador, y
`X_test`, los clientes que el modelo nunca vio.

**Por qué hoy se mira prueba más de una vez:** el modelo ya quedó elegido y congelado en la sesión 8.
Explicarlo sobre prueba es la **auditoría final** antes de entregarlo a Cobranza; nada de lo que se vea
se usa para cambiarlo. Si un hallazgo llevara a ajustar el modelo, ese ajuste se volvería a validar en
`X_trainval`, no acá.

<!-- contrato: accion=partir -->
"""),
    code("""
from sklearn.model_selection import train_test_split  # sortea la partición

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
display(Markdown(f'`X_trainval`: **{num(len(X_trainval))}** clientes · `X_test`: **{num(len(X_test))}**.'))
"""),
    md("""
### El ganador de la sesión 8, reentrenado

El bosque con los hiperparámetros que eligió la búsqueda de la sesión 8: `max_features='sqrt'` y
`min_samples_leaf=20`. La preparación es la misma, con un cambio que no altera el ajuste: sale como
**tabla con nombres de columna** (`set_output(transform='pandas')`), para que SHAP pueda rotular cada
variable en sus gráficos.

<!-- contrato: accion=reentrenar-ganador -->
"""),
    code("""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

preparar = ColumnTransformer([
    ('num', 'passthrough', COLUMNAS_NUM),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), COLUMNAS_CAT),
], verbose_feature_names_out=False).set_output(transform='pandas')
ganador = Pipeline([('preparar', preparar),
                    ('modelo', RandomForestClassifier(max_features='sqrt', min_samples_leaf=20,
                                                      random_state=42, n_jobs=-1))])
ganador.fit(X_trainval, y_trainval)
"""),
    code("""
prob_test = ganador.predict_proba(X_test)[:, 1]   # probabilidad de impago de cada cliente de prueba
display(Markdown(f'ROC-AUC en prueba: **{num(roc_auc_score(y_test, prob_test), 3)}**, la cifra que obtuvo '
                 f'el bosque al abrir la prueba en la sesión 8. Es el mismo modelo.'))
"""),
    md("""
# Tiempo 1 · Yo lo hago

## Lo que la importancia nativa no dice

El bosque trae su propia **importancia de variables** (`feature_importances_`): cuánto ayudó cada
columna a separar a quienes pagan de quienes no, sumado sobre todos los cortes de todos los árboles.

<!-- contrato: accion=importancia-nativa -->
"""),
    code("""
nativa = pd.Series(ganador.named_steps['modelo'].feature_importances_,
                   index=ganador.named_steps['preparar'].get_feature_names_out())
nativa.sort_values(ascending=False).head(5).round(3)
"""),
    md("""
Son todos números positivos y uno por variable. No dicen **hacia dónde** empuja cada variable (¿más
mora sube o baja el riesgo?) ni **cuánto** pesó en un cliente concreto. Para Cobranza, que pregunta por
**un** cliente, no alcanza.
"""),
    md("""
## Abrir el modelo: `shap.TreeExplainer`

**SHAP** reparte la probabilidad predicha de cada cliente entre sus variables: parte de un **valor base**
y le suma o resta el **aporte** de cada variable. `shap.TreeExplainer` es la versión para modelos de
árboles; recibe el bosque ya entrenado (el paso `modelo` del pipeline) y se aplica a los clientes
**ya preparados**.

El resultado tiene tres dimensiones: **cliente × variable × clase**. El bosque entrega una probabilidad
para cada clase, así que con `[:, :, 1]` nos quedamos con la de **impago**.

<!-- contrato: accion=explicar -->
"""),
    code("""
import shap  # la librería de explicaciones

X_test_prep = ganador.named_steps['preparar'].transform(X_test)   # los clientes de prueba, preparados
explicador = shap.TreeExplainer(ganador.named_steps['modelo'])
explicacion = explicador(X_test_prep)
impago = explicacion[:, :, 1]   # todos los clientes, todas las variables, la clase impago
"""),
    md("""
### La comprobación: base + aportes = probabilidad

Para **cada** cliente, el valor base más la suma de sus aportes tiene que dar exactamente la probabilidad
que entrega el modelo. La celda calcula la mayor diferencia entre ambas en los clientes de prueba.

<!-- contrato: accion=comprobar-aditividad -->
"""),
    code("""
import numpy as np

base = impago.base_values[0]
reconstruida = base + impago.values.sum(axis=1)       # base + suma de los aportes, cliente por cliente
error_aditividad = np.abs(reconstruida - prob_test).max()
print('mayor diferencia:', error_aditividad, '· menor que 1e-12:', error_aditividad < 1e-12)
display(Markdown(
    f'El valor base es **{num(100 * base, 2)} %**: el promedio, entre los árboles, de la proporción de '
    f'impago que vio cada uno en su remuestra de entrenamiento. Queda muy cerca de la proporción de impago de `y_trainval`, '
    f'{num(100 * y_trainval.mean(), 2)} %, sin ser idéntico, porque cada árbol vio una remuestra distinta. '
    f'La diferencia máxima entre base + aportes y la probabilidad es del orden de {error_aditividad:.0e}: '
    f'error de redondeo del computador, o sea, **cero**.'
))
"""),
    md("""
## Explicación global: el enjambre, con `shap.plots.beeswarm`

`shap.plots.beeswarm` dibuja **un punto por cliente y por variable**. Se lee en tres pasos:

1. **Magnitud:** las variables van ordenadas de arriba abajo por su aporte promedio, sin signo.
2. **Dirección:** a la derecha del 0, la variable **subió** la probabilidad de impago; a la izquierda, la
   bajó. El color es el valor de la variable: rojo alto, azul bajo.
3. **Dispersión:** una nube ancha dice que la misma variable pesa muy distinto según el cliente.

<!-- contrato: accion=enjambre -->
"""),
    code("""
shap.plots.beeswarm(impago)
"""),
    code("""
promedio = pd.Series(np.abs(impago.values).mean(axis=0), index=impago.feature_names)
promedio = promedio.sort_values(ascending=False)
display(Markdown(
    f'Las cuatro primeras: `{promedio.index[0]}`, `{promedio.index[1]}`, `{promedio.index[2]}` y '
    f'`{promedio.index[3]}`. La primera mueve, en promedio, **{num(100 * promedio.iloc[0], 1)} puntos** la '
    f'probabilidad de un cliente; la cuarta, {num(100 * promedio.iloc[3], 1)}. Las tres de mora '
    f'están correlacionadas y se reparten el crédito: se leen como un bloque de historial de pago.'
))
"""),
    md("""
# Tiempo 2 · Lo hacemos juntos

## Explicación local: dos clientes contrastantes

El enjambre resume a todos. Cobranza pregunta por **uno**. Tomamos el cliente de prueba con la
probabilidad más alta y el de la más baja: `argmax` y `argmin` devuelven su **posición** en `X_test`.

<!-- contrato: accion=elegir-contrastes -->
"""),
    code("""
alto = int(np.argmax(prob_test))   # posición del cliente con más riesgo
bajo = int(np.argmin(prob_test))   # posición del cliente con menos riesgo
id_alto = datos.loc[X_test.index[alto], 'id_cliente']
id_bajo = datos.loc[X_test.index[bajo], 'id_cliente']
display(Markdown(f'**{id_alto}**: {num(100 * prob_test[alto], 1)} % · **{id_bajo}**: '
                 f'{num(100 * prob_test[bajo], 1)} %.'))
"""),
    md("""
### Primero, la tabla de descomposición

Antes del gráfico, los números: el valor de cada variable del cliente de más riesgo y su aporte en
puntos, ordenados por tamaño sin importar el signo.

<!-- contrato: accion=tabla-descomposicion -->
"""),
    code("""
descomposicion = pd.DataFrame({'valor': X_test_prep.iloc[alto],
                               'aporte (puntos)': 100 * impago.values[alto]})
descomposicion = descomposicion.sort_values('aporte (puntos)', key=abs, ascending=False)
descomposicion.head(6).round(1)
"""),
    code("""
moras = descomposicion.loc[['meses_mora', 'meses_mora_acumulados', 'meses_mora_previo'], 'aporte (puntos)']
display(Markdown(
    f'{id_alto} parte de la base, {num(100 * base, 1)} %, y sus aportes suman '
    f'**{pts(prob_test[alto] - base)} puntos**: llega a {num(100 * prob_test[alto], 1)} %. De esos puntos, '
    f'**{pts(moras.sum() / 100)}** vienen de las tres variables de mora.'
))
"""),
    md("""
### La cascada: `shap.plots.waterfall`

`shap.plots.waterfall` dibuja esa misma tabla como una **cascada**: abajo, el valor base (`E[f(X)]`);
cada barra suma (rojo) o resta (azul) el aporte de una variable, y arriba queda la probabilidad del
cliente (`f(x)`). El gráfico escribe las cifras en **proporción, con punto decimal y dos decimales**:
`+0.27` son unos 27 puntos (26,6 en la tabla anterior). Las variables que pesan poco se agrupan en la
barra de más abajo, justo sobre el valor base.

<!-- contrato: accion=cascada-contraste -->
"""),
    code("""
shap.plots.waterfall(impago[alto])
"""),
    md("""
Ahora el cliente de menos riesgo. Antes de ejecutar: ¿qué variables esperan ver empujando hacia la
izquierda?
"""),
    code("""
shap.plots.waterfall(impago[bajo])
"""),
    code("""
display(Markdown(
    f'{id_bajo} parte de la misma base y sus aportes suman **{pts(prob_test[bajo] - base)} puntos**. Las '
    f'mismas variables, con otros valores, empujan hacia el otro lado: por eso la explicación global no '
    f'reemplaza a la local.'
))
"""),
    md("""
## El gráfico de dependencia: `shap.plots.scatter`

`shap.plots.scatter` toma **una** variable y pone a cada cliente de prueba como un punto: en el eje
horizontal, su valor de `meses_mora`; en el vertical, el aporte que esa variable le dio. Con
`color=impago`, SHAP colorea los puntos con la **segunda variable** que más parece explicar la
dispersión vertical; su nombre aparece en la barra de color.

<!-- contrato: accion=dependencia -->
"""),
    code("""
shap.plots.scatter(impago[:, 'meses_mora'], color=impago)
"""),
    md("""
### El punto de quiebre, calculado

El aporte promedio de `meses_mora` en cada nivel, con cuántos clientes de prueba hay en cada uno.

<!-- contrato: accion=punto-quiebre -->
"""),
    code("""
por_nivel = pd.DataFrame({'meses_mora': X_test_prep['meses_mora'],
                          'aporte (puntos)': 100 * impago[:, 'meses_mora'].values})
por_nivel = por_nivel.groupby('meses_mora')['aporte (puntos)'].agg(['mean', 'count'])
por_nivel.columns = ['aporte medio (puntos)', 'clientes']
por_nivel.round(1)
"""),
    md("""
¿Entre qué niveles cambia más el aporte? `diff` resta cada fila de la anterior.
"""),
    code("""
saltos = por_nivel['aporte medio (puntos)'].diff()   # cuánto cambia el aporte de un nivel al siguiente
mayor = saltos.idxmax()
pocos = por_nivel[por_nivel['clientes'] < 30].index
display(Markdown(
    f'El mayor salto ocurre al pasar a **{mayor} meses** de mora: **+{num(saltos.max(), 1)} puntos** de '
    f'aporte promedio. Los niveles con menos de 30 clientes ({", ".join(map(str, pocos))}) se leen con '
    f'cautela: su promedio descansa en pocos casos (es una heurística de tamaño, no un umbral exacto).'
))
"""),
    md("""
**Lo que este gráfico no dice.** Muestra cómo se asocian, en estos clientes, los meses de mora con lo
que el modelo predice. No dice qué pasaría si Cobranza **cambiara** la mora de un cliente: el modelo
aprendió que el atraso acompaña al impago, no qué ocurre si se elimina el atraso.
"""),
    md("""
# Tiempo 3 · Lo hacen ustedes

## La actividad: auditar los peores errores

Los errores más informativos son los que el modelo cometió **con alta confianza**:

- **Falsas alarmas confiadas:** clientes que **pagaron** y a los que el modelo dio la probabilidad de
  impago más alta.
- **Incumplimientos no vistos:** clientes que **no pagaron** y a los que el modelo dio la probabilidad
  más baja.

1. **Ejecuten** la celda que aísla los 10 peores de cada tipo y lean las dos tablas.
2. **Miren la cascada** del peor caso de cada tipo y comparen qué variables lo empujaron.
3. **Completen la tabla diagnóstica** del final, con sus palabras.

**Criterio de éxito:** cada hipótesis nombra información que **el archivo no tiene** y que podría
explicar el error, y la última columna dice qué **no** se puede afirmar con estos datos.

**Evidencia observable:** las dos tablas, las dos cascadas y la tabla diagnóstica completa; el
**feedback formativo** llega en la puesta en común.

### Paso 1 · Los 10 peores de cada tipo

La tabla ya viene filtrada y ordenada: el trabajo es leerla, no construirla. La columna `posicion` es
el lugar del cliente en `X_test`, el que se usa para pedir su cascada.

<!-- contrato: accion=aislar-peores -->
"""),
    code("""
MORAS = ['meses_mora', 'meses_mora_previo', 'meses_mora_acumulados']
resultado = datos.loc[X_test.index, ['id_cliente'] + MORAS + ['limite_credito', 'uso_credito']]
resultado['probabilidad'] = prob_test.round(3)
resultado['incumplio'] = y_test
resultado['posicion'] = np.arange(len(resultado))
"""),
    code("""
pagaron = resultado[resultado['incumplio'] == 0]
falsas_alarmas = pagaron.sort_values('probabilidad', ascending=False).head(10)
falsas_alarmas
"""),
    md("""
Y los que **no pagaron**, ordenados de la probabilidad más baja a la más alta: arriba, los que el
modelo menos esperaba.
"""),
    code("""
no_pagaron = resultado[resultado['incumplio'] == 1]
no_vistos = no_pagaron.sort_values('probabilidad').head(10)
no_vistos
"""),
    md("""
**Preguntas guía:** ¿qué tienen en común las 10 falsas alarmas en las columnas de mora? ¿Y los 10
incumplimientos no vistos? ¿Qué le faltaría saber al modelo para no equivocarse con ellos?

### Paso 2 · La cascada del peor caso de cada tipo

`iloc[0]` toma la primera fila de cada tabla, la peor. Cambien el 0 por otro número para mirar otro caso.

<!-- contrato: accion=cascada-error -->
"""),
    code("""
caso = falsas_alarmas['posicion'].iloc[0]   # la falsa alarma más confiada
shap.plots.waterfall(impago[caso])
"""),
    md("""
El incumplimiento menos esperado. Comparen: ¿qué variables lo dejaron tan abajo de la base?
"""),
    code("""
caso = no_vistos['posicion'].iloc[0]        # el incumplimiento menos esperado
shap.plots.waterfall(impago[caso])
"""),
    md("""
### Paso 3 · La tabla diagnóstica

Hagan doble clic en esta celda para completarla. Una hipótesis es una **información ausente**, no una
causa demostrada.

| Tipo de error | Qué comparten (dato de las tablas) | Hipótesis: qué información falta en el archivo | Qué NO podemos afirmar |
|:--|:--|:--|:--|
| Falsas alarmas confiadas |  |  |  |
| Incumplimientos no vistos |  |  |  |

**Límite del modelo, en una frase:**
"""),
    md("""
# Cierre

## Lo que le podemos responder a Cobranza, y lo que no

- **Sí:** descomponer con exactitud por qué el modelo marcó a un cliente, partiendo del valor base y
  sumando el aporte de cada variable, y usar esa probabilidad para **priorizar** a quién llamar primero.
- **No:** prometer que cambiar una variable —condonar la cuota atrasada, subir el límite— bajará el
  riesgo real. SHAP explica **al modelo**, no al mundo; saber qué pasa al intervenir exige un
  experimento.
- **Y sus errores marcan su frontera:** el modelo no puede anticipar lo que sus variables no registran.
"""),
    md(f"""
## Atribución de datos

- **Creador:** Yeh, I. C., & Lien, C. H. (2009)
- **Fuente:** [UCI Machine Learning Repository — *Default of Credit Card Clients*, dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** obra derivada. Se tomaron 6.000 registros y 16 columnas del original de 30.000 ×
  25, con los nombres traducidos al español en `snake_case`, la escolaridad recodificada a etiquetas y
  dos razones derivadas nuevas. Los
  atributos sexo y estado civil se excluyeron deliberadamente.
"""),
]


def _arbol(fuente: str) -> ast.AST:
    """AST de una celda, con las magias de línea neutralizadas como hace el checker."""
    limpio = "\n".join(l for l in fuente.splitlines() if not l.lstrip().startswith(("%", "!")))
    return ast.parse(limpio)


def llamada(fuente: str, nombre: str) -> dict:
    """Cuenta las llamadas a `nombre` y su cantidad de argumentos, por AST. Falla cerrado si difieren."""
    firmas = [
        len(nodo.args) + len(nodo.keywords)
        for nodo in ast.walk(_arbol(fuente))
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name) and nodo.func.id == nombre
    ]
    assert firmas, f"no hay llamadas a {nombre}: el contrato no puede declararlas"
    distintas = set(firmas)
    assert len(distintas) == 1, (
        f"las llamadas a {nombre} tienen {sorted(distintas)} argumentos; un solo número no las describe")
    return {"funcion": nombre, "argumentos": firmas[0], "cantidad": len(firmas)}


def sin_comprensiones(codigo: list[dict]) -> None:
    """Python básico (docente, 2026-09-09): el material del estudiante no usa comprensiones ni lambda."""
    prohibidos = (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Lambda)
    for celda in codigo:
        for nodo in ast.walk(_arbol("".join(celda["source"]))):
            assert not isinstance(nodo, prohibidos), (
                f"una celda usa {type(nodo).__name__}: {''.join(celda['source'])[:80]!r}")


def contrato(celdas: list[dict]) -> dict:
    """Deriva `metadata.curso_contrato` de la lista FINAL de celdas."""
    codigo = [c for c in celdas if c["cell_type"] == "code"]
    markdown = sum(1 for c in celdas if c["cell_type"] == "markdown")
    assert markdown >= len(codigo), (
        f"{markdown} celdas markdown contra {len(codigo)} de código: el cuaderno es material de "
        "estudio y la explicación va en Markdown")
    sin_comprensiones(codigo)
    fuente = "\n".join("".join(c["source"]) for c in codigo)
    lineas_por_celda = [
        len([l for l in "".join(c["source"]).splitlines()
             if l.strip() and not l.strip().startswith("#")])
        for c in codigo
    ]
    return {
        "forma": {
            "markdown": markdown,
            "codigo": len(codigo),
            "llamadas": [llamada(fuente, "train_test_split"), llamada(fuente, "pts")],
        },
        "acciones": [
            {"marcador": "fijar-base", "asignacion": "datos"},
            {"marcador": "declarar-positiva", "asignacion": "y"},
            {"marcador": "partir", "codigo": "train_test_split"},
            {"marcador": "reentrenar-ganador", "codigo": "RandomForestClassifier"},
            {"marcador": "importancia-nativa", "asignacion": "nativa"},
            {"marcador": "explicar", "codigo": "TreeExplainer"},
            {"marcador": "comprobar-aditividad", "asignacion": "error_aditividad"},
            {"marcador": "enjambre", "codigo": "shap.plots.beeswarm"},
            {"marcador": "elegir-contrastes", "asignacion": "alto"},
            {"marcador": "tabla-descomposicion", "asignacion": "descomposicion"},
            {"marcador": "cascada-contraste", "codigo": "shap.plots.waterfall"},
            {"marcador": "dependencia", "codigo": "shap.plots.scatter"},
            {"marcador": "punto-quiebre", "asignacion": "por_nivel"},
            {"marcador": "aislar-peores", "asignacion": "resultado"},
            {"marcador": "cascada-error", "codigo": "shap.plots.waterfall"},
        ],
        # Es una MEDIANA, no un máximo por celda. Se fija al valor real medido, con un margen de 1.
        "max_mediana_lineas_codigo": int(statistics.median(lineas_por_celda)) + 1,
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
    c = nb["metadata"]["curso_contrato"]
    print(f"escrito {SALIDA.relative_to(REPO_ROOT)}")
    print(f"  {len(celdas)} celdas · {c['forma']['markdown']} markdown · {c['forma']['codigo']} código")
    print(f"  contrato: mediana de líneas {c['max_mediana_lineas_codigo'] - 1} · llamadas {c['forma']['llamadas']}")


if __name__ == "__main__":
    main()
