#!/usr/bin/env python3
"""Construye notebooks/07-clasificacion.ipynb.

POR QUÉ EXISTE UN GENERADOR: el `.ipynb` no se edita a mano. Si hay que corregir algo, se corrige acá y
se vuelve a ejecutar con `make nb-build C=07`.

ESTRUCTURA: los tres tiempos de la `practica` del temario, más el cierre. (1) El docente pone el
BASELINE primero —recuperación de la sesión 1, no materia nueva—, cambia la última pieza del flujo por
`LogisticRegression`, separa `predict` de `predict_proba` y lee dos coeficientes como odds ratio.
(2) Entre todos: la matriz con el umbral en 0,5, la accuracy contra el baseline, precision y recall,
el barrido del umbral, la cuenta del costo con razón 5 a 1 y la coda de ROC-AUC. (3) La actividad:
cada grupo elige una posición de gestión (cobranza 4 a 1, comercial 1,5 a 1), elige el umbral sobre
validación, abre prueba UNA vez y defiende su lista en tres líneas. Cierre: la respuesta a la pregunta
de la clase con el criterio, sin presuponer la posición de cada grupo.

LA MEDICIÓN ES LA DEL GENERADOR DE FIGURAS (`assets/ilustraciones/_build_c07_figuras.py`): mismo
archivo, mismas predictoras en el mismo orden, mismo pipeline, partición 60/20/20 estratificada con la
misma semilla y el mismo orden de `train_test_split`. Por eso las cifras del cuaderno coinciden con las
del deck sin copiarse: se RECALCULAN en cada celda y se muestran con Markdown dinámico por f-string.
Las SVG embebidas salen de esa misma medición.

PRUEBA SE ABRE UNA SOLA VEZ, en el paso 3 de la actividad. Todo lo que el cuaderno explora —barrido,
matriz, métricas, costo, ROC-AUC— sale de validación. El tamaño de prueba se informa como lo que quedó
fuera de `X_trainval`, sin leer `X_test`. El contrato lo protege con `max_celdas_lectura`, derivado por
AST con la misma regla del checker y comprobado aquí con un `assert`: si una celda más lee prueba, el
generador falla antes de escribir.

SIN REAJUSTE ANTES DE PRUEBA, a diferencia de la sesión 6: el umbral se eligió para las probabilidades
de ESTE modelo, y uno reajustado con entrenamiento más validación entregaría otras. Es también lo que
hace el generador de figuras, de donde salen el 18 % y el 6 % que cita el deck.

UN SOLO PIPELINE CON PREPARACIÓN: el `ColumnTransformer` se construye dentro del `Pipeline`, no en una
variable compartida (ver `diseno-clase`, notebooks). El baseline no lleva preparación porque
`DummyClassifier` ignora las predictoras, y la logística de dos predictoras va sin escalar a propósito.

FRONTERAS, heredadas del `alcance_no` y verificadas al escribir:
  · Ni árboles ni validación cruzada: es la sesión 8, que se nombra solo como siguiente paso.
  · Ni curva precision-recall, ni `class_weight`, ni calibración desarrollada (se nombra en una frase).
  · Ni la fórmula del costo esperado ni la regla del cociente de costos: el umbral se MIDE.
  · ROC-AUC solo como coda, con su límite dicho.
  · Ningún coeficiente se lee como efecto causal.

Uso:
    make nb-build C=07   (regenera, ejecuta en sandbox y deja el ejecutado publicado)
    make nb-check C=07   (verifica el contrato)
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
SALIDA = AQUI / "07-clasificacion.ipynb"
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
<p>Semana 07: Clasificación — probabilidades, umbrales y costos</p>
</div>
</div>
"""),
    md("""
# 07 · Clasificación: probabilidades, umbrales y costos

> **La pregunta de hoy:** ¿a qué clientes debe llamar cobranza antes de que dejen de pagar, y cómo
> defendemos esa lista?

Tres tiempos: **yo lo hago** (baseline y regresión logística), **lo hacemos juntos** (matriz, umbral
y costo) y **lo hacen ustedes** (elegir un umbral y defenderlo).
"""),
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas; la celda queda para correr fuera de Colab. La siguiente
define `num`, que escribe las cifras con punto de miles y coma decimal.
"""),
    code("""
%%capture
!pip install -q pandas numpy scikit-learn matplotlib
"""),
    code("""
from IPython.display import Markdown, display  # muestra frases con las cifras ya calculadas

def num(x, dec=0):
    # miles con punto, decimales con coma y el menos tipográfico
    texto = f'{x:,.{dec}f}'.replace(',', '@').replace('.', ',').replace('@', '.')
    return texto.replace('-', '−')
"""),
    md("""
## Cargar datos

Una cartera de **tarjetas de crédito**, `riesgo_impago.csv`, leída con **pandas**.

<!-- contrato: accion=fijar-base -->
"""),
    code("""
import os                # revisa si el archivo está en una carpeta local
import pandas as pd      # la librería de tablas: lee el archivo y lo deja como DataFrame

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

datos = pd.read_csv(BASE + 'riesgo_impago.csv')
print('registros y columnas:', datos.shape)
datos.head()
"""),
    md("""
## El encuadre, antes de medir nada

- **Unidad de observación:** un registro es un **cliente observado en un mes**.
- **Objetivo:** si el cliente **incumple el pago del mes siguiente** (`incumplio_pago`, `si` o `no`).
  Es una categoría, no una cantidad: esto es **clasificación**, binaria porque hay dos categorías;
  con más sería multiclase, con la misma lógica.
- **Momento de la predicción:** el cierre del mes observado; solo vale lo anterior a él.
- **Clase positiva:** **incumplir**, el evento que interesa detectar. Todas las métricas de hoy se
  calculan respecto de ella.

La celda codifica el objetivo (1 = incumple, 0 = paga) y deja como predictoras todas las columnas
menos `id_cliente` y el objetivo.

<!-- contrato: accion=declarar-positiva -->
"""),
    code("""
y = datos['incumplio_pago'].map({'no': 0, 'si': 1})  # 1 = incumple, la clase positiva

# nivel_educacion es la única categórica: se prepara distinto que las numéricas
COLUMNAS_CAT = ['nivel_educacion']
COLUMNAS_NUM = []
for columna in datos.columns:  # todas menos el identificador, el objetivo y la categórica
    if columna not in ['id_cliente', 'incumplio_pago'] + COLUMNAS_CAT:
        COLUMNAS_NUM.append(columna)
X = datos[COLUMNAS_NUM + COLUMNAS_CAT]

display(Markdown(
    f'Quedan **{len(COLUMNAS_NUM)}** predictoras numéricas y **{len(COLUMNAS_CAT)}** categórica.'
))
"""),
    md("""
### ¿Hay desbalance de clases? Se mide con `value_counts`

Antes de medir cualquier modelo, hay que contar cuántos registros tiene cada clase. `value_counts`
cuenta los registros por valor, y con `normalize=True` entrega la misma cuenta como proporción.
"""),
    code("""
conteo = datos['incumplio_pago'].value_counts()                     # registros por clase
proporcion = datos['incumplio_pago'].value_counts(normalize=True)   # la misma cuenta, como proporción
pd.DataFrame({'registros': conteo, 'proporción': proporcion.round(3)})
"""),
    code("""
display(Markdown(
    f'Incumple el **{num(100 * proporcion["si"], 1)} %** de los {num(conteo.sum())} registros: '
    f'**{num(conteo["si"])}** contra **{num(conteo["no"])}** que pagan. Hay **desbalance de clases**, '
    f'y la clase positiva, la que interesa detectar, es la **minoritaria**.'
))
"""),
    md("""
### La partición: 60 / 20 / 20, estratificada

Es la de la sesión 6: entrenamiento ajusta, validación decide —hoy, el **umbral**— y prueba se abre
una sola vez, en la actividad. `stratify=y` hace que cada pedazo conserve la proporción de
incumplidores. La figura muestra por qué, con 25 clientes de juguete: al azar, validación puede
quedar sin un solo incumplidor, y no habría a quién detectar.

""" + figura_base64(
        "c07-estratificar.svg",
        "Ilustración con 25 clientes, 5 de los cuales incumplen, dibujados como círculos azules y "
        "bermellón, repartidos en entrenamiento, validación y prueba de dos maneras. Al azar, "
        "validación queda con 0 de 5 incumplidores. Estratificado, las tres partes quedan con el 20 "
        "por ciento.") + """

<!-- contrato: accion=partir-en-tres -->
"""),
    code("""
from sklearn.model_selection import train_test_split  # sortea la partición

# 20 % a prueba, y del resto un 25 % a validación: quedan 60 / 20 / 20
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)

display(Markdown(
    f'Entrenamiento: **{num(len(X_train))}** · validación: **{num(len(X_val))}** · prueba: '
    f'**{num(len(datos) - len(X_trainval))}**, guardada hasta la actividad. En validación incumplen '
    f'**{num(y_val.sum())}** clientes, el {num(100 * y_val.mean(), 1)} %; en el archivo completo, '
    f'el {num(100 * y.mean(), 1)} %.'
))
"""),
    md("""
# Tiempo 1 · Yo lo hago

## El baseline: suponer que todos pagan

`DummyClassifier` con `strategy='most_frequent'` ignora las predictoras y responde siempre la clase
más frecuente, «paga»: la regla de la sesión 1. Lo medimos sobre validación con la *accuracy*
(exactitud), la proporción de clientes bien clasificados que calcula `accuracy_score`, y con cuántos
incumplidores detecta.

<!-- contrato: accion=el-baseline -->
"""),
    code("""
from sklearn.dummy import DummyClassifier   # el modelo que responde siempre lo mismo
from sklearn.metrics import accuracy_score  # la proporción de clientes bien clasificados

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_val)
accuracy_baseline = accuracy_score(y_val, pred_baseline)
detectados_baseline = ((pred_baseline == 1) & (y_val == 1)).sum()

display(Markdown(
    f'El baseline acierta el **{num(100 * accuracy_baseline, 1)} %** de los {num(len(y_val))} '
    f'clientes de validación, justo la proporción de los que pagan, y detecta '
    f'**{num(detectados_baseline)}** de los **{num(y_val.sum())}** que incumplen. Esa accuracy alta '
    f'mide el **desbalance de clases** —una mucho más frecuente que la otra—, no un modelo.'
))
"""),
    md("""
## La última pieza del flujo: `LogisticRegression`

Una regresión lineal sobre 0 y 1 puede predecir más de 1, y una probabilidad tiene techo. La
**regresión logística** calcula en dos pasos: **suma** como la lineal,
z = β<sub>0</sub> + β<sub>1</sub>·x<sub>1</sub> + … + β<sub>p</sub>·x<sub>p</sub>, y una **curva en
S** convierte esa suma en probabilidad, p = 1 / (1 + e<sup>−z</sup>), que siempre queda entre 0 y 1.
Aprende por **máxima verosimilitud** (*maximum likelihood*): busca los coeficientes que dan
probabilidad alta a quienes incumplieron y baja a quienes pagaron.

En el `Pipeline` de las sesiones 5 y 6 solo cambia el paso `modelo`. `LogisticRegression`
**regulariza por defecto**, como Ridge, y por eso las numéricas se escalan; `max_iter=2000` le da
vueltas suficientes para converger.

<!-- contrato: accion=el-modelo -->
"""),
    code("""
from sklearn.pipeline import Pipeline             # encadena la preparación y el modelo
from sklearn.compose import ColumnTransformer     # prepara cada grupo de columnas a su manera
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression  # la regresión logística

modelo = Pipeline([
    ('preparar', ColumnTransformer([
        ('num', StandardScaler(), COLUMNAS_NUM),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUMNAS_CAT),
    ])),
    ('modelo', LogisticRegression(max_iter=2000)),  # la única pieza que cambia
])
modelo.fit(X_train, y_train)
"""),
    md("""
## `predict` da una etiqueta; `predict_proba`, un número

`predict` devuelve 0 (paga) o 1 (incumple) con un umbral interno de 0,5. `predict_proba` devuelve una
columna de probabilidad por clase, en el orden de `modelo.classes_`; la de incumplir es la segunda,
`[:, 1]`. Para gestión se quiere la probabilidad: deja la decisión a quien conoce el costo de cada
error.

<!-- contrato: accion=la-probabilidad -->
"""),
    code("""
print('orden de las columnas de predict_proba:', modelo.classes_)
prob_val = modelo.predict_proba(X_val)[:, 1]   # la probabilidad de incumplir, cliente por cliente

clientes_val = pd.DataFrame({'id_cliente': datos.loc[X_val.index, 'id_cliente'],
                             'probabilidad': prob_val,
                             'predict': modelo.predict(X_val)})
display(clientes_val.head().round(3))

iguales = ((prob_val >= 0.5) == clientes_val['predict']).sum()
display(Markdown(
    f'En **{num(iguales)}** de los **{num(len(clientes_val))}** clientes de validación, `predict` '
    f'coincide con preguntar si la probabilidad llega a 0,5: **`predict` usa un 0,5 que nadie eligió**.'
))
"""),
    md("""
### Cuatro clientes, su suma z y su probabilidad

`decision_function` devuelve la suma z de cada cliente, antes de la curva en S; `predict_proba` es esa
suma pasada por la curva. Cuatro clientes de validación, con tres de las catorce predictoras:
"""),
    code("""
cuatro = ['R-2911', 'R-1637', 'R-3258', 'R-0823']        # los clientes marcados en la figura
en_validacion = datos.loc[X_val.index]
marcados = en_validacion['id_cliente'].isin(cuatro).to_numpy()
tabla_z = en_validacion.loc[marcados, ['id_cliente', 'meses_mora', 'uso_credito', 'cobertura_pago',
                                       'incumplio_pago']]
tabla_z['z'] = modelo.decision_function(X_val[marcados])  # la suma, antes de la curva en S
tabla_z['probabilidad'] = prob_val[marcados]              # la suma, después de la curva
tabla_z.sort_values('z').round(4)   # cuatro decimales: R-3258 queda justo bajo 0,5
"""),
    md(figura_base64(
        "c07-logistica-real.svg",
        "Regresión logística ajustada a la cartera: en el eje horizontal, z, la suma del modelo con las "
        "14 predictoras; en el vertical, la probabilidad de incumplir. La curva en S pasa por 0,5 cuando "
        "z vale 0. Cada cliente de validación es un punto en su z: arriba, en 1, los que incumplieron; "
        "abajo, en 0, los que pagaron. Los cuatro clientes de la tabla están marcados con su "
        "identificador y con su probabilidad sobre la curva. A la derecha predominan los de arriba y a "
        "la izquierda los de abajo, pero en el medio se mezclan.")
       + """

Cada punto es un cliente de validación en su z: arriba, los que incumplieron; abajo, los que pagaron.
A más suma, más probabilidad, pero `R-1637` incumplió con una suma baja: en el medio las clases se
mezclan. Por eso la probabilidad sirve para **ordenar** a quién llamar primero si el equipo alcanza
para pocos. Ordenar no es acertar la frecuencia: un 0,9 del modelo no asegura 9 de cada 10.

Los primeros de la lista:
"""),
    code("""
clientes_val.sort_values('probabilidad', ascending=False).head().round(3)
"""),
    md("""
## Un coeficiente se lee como odds ratio

Los **odds** dividen la probabilidad de que algo ocurra por la de que no ocurra, p / (1 − p): 0,75
son odds de 3, «tres a uno». Un coeficiente se comunica como **odds ratio**, e<sup>β</sup>: cuánto se
multiplican los odds al subir la variable en una unidad, manteniendo las demás iguales; **1** es sin
efecto.

Para leerlo en unidades del negocio ajustamos una **segunda logística**, sin escalar, con dos
predictoras: `meses_mora`, meses de atraso, y `uso_credito`, lo facturado como fracción del cupo.
`coef_` guarda sus coeficientes y `np.exp`, de **NumPy**, calcula la exponencial.

<!-- contrato: accion=odds-ratio -->
"""),
    code("""
import numpy as np  # cálculo numérico: acá, la exponencial

legibles = ['meses_mora', 'uso_credito']
dos_predictoras = LogisticRegression(max_iter=2000).fit(X_train[legibles], y_train)
odds_ratio = pd.DataFrame({'odds ratio': np.exp(dos_predictoras.coef_[0])}, index=legibles)
display(odds_ratio.round(2))

mora = odds_ratio.loc['meses_mora', 'odds ratio']
uso = odds_ratio.loc['uso_credito', 'odds ratio']
display(Markdown(
    f'Cada mes más de mora, **manteniendo igual el uso del cupo**, multiplica los odds de incumplir '
    f'por **{num(mora, 2)}**. Usar el cupo entero en vez de nada, **manteniendo igual la mora**, los '
    f'multiplica por **{num(uso, 2)}**. Se multiplican los odds, no la probabilidad, y es una '
    f'**asociación**, no un efecto.'
))
"""),
    md("""
### Una probabilidad, calculada paso a paso

La misma cuenta, a mano, para el cliente de validación `R-0253`, con los coeficientes de esa
logística redondeados a dos decimales para poder repetirla con calculadora:
"""),
    code("""
cliente = en_validacion.loc[en_validacion['id_cliente'] == 'R-0253', legibles]  # un cliente de validación
b0 = round(dos_predictoras.intercept_[0], 2)          # el término constante, a dos decimales
b_mora, b_uso = dos_predictoras.coef_[0].round(2)     # un coeficiente por predictora, a dos decimales
mora_c, uso_c = cliente.iloc[0].round(2)

z = b0 + b_mora * mora_c + b_uso * uso_c  # paso 1: la suma
p_mano = 1 / (1 + np.exp(-z))             # paso 2: la curva en S
p_modelo = dos_predictoras.predict_proba(cliente)[0, 1]
display(Markdown(
    f'Cliente `R-0253`: z = {num(b0, 2)} + {num(b_mora, 2)} × {num(mora_c)} + {num(b_uso, 2)} × '
    f'{num(uso_c, 2)} = **{num(z, 2)}**, y p = 1 / (1 + e<sup>−{num(z, 2)}</sup>) = **{num(p_mano, 2)}**. '
    f'`predict_proba` da {num(p_modelo, 2)}: es la misma cuenta. Con las catorce predictoras es igual, '
    f'con más sumandos.'
))
"""),
    md("""
# Tiempo 2 · Lo hacemos juntos

## La matriz de confusión, con el umbral en 0,5

La lista de llamadas sale de un **umbral**: entra quien lo alcanza, y con otro umbral la matriz
cambia. `confusion_matrix` recibe lo observado y lo predicho: las filas son lo que pasó y las columnas
lo predicho, `labels=[0, 1]` fija el orden de las clases, y `.ravel()` entrega verdaderos negativos (VN), falsos positivos
(FP), falsos negativos (FN) y verdaderos positivos (VP). Un FP es una **falsa alarma**, una llamada de
más; un FN, un **impago que se escapa**. No cuestan lo mismo. `ConfusionMatrixDisplay` la dibuja con
los mismos argumentos, y la diagonal da la accuracy, que comparamos con la del baseline.

<!-- contrato: accion=la-matriz -->
"""),
    code("""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # cuenta y dibuja los cuatro resultados
import matplotlib.pyplot as plt

UMBRAL_DEFECTO = 0.5
lista_05 = np.where(prob_val >= UMBRAL_DEFECTO, 1, 0)   # 1 = entra en la lista de llamadas
vn, fp, fn, vp = confusion_matrix(y_val, lista_05, labels=[0, 1]).ravel()
matriz = ConfusionMatrixDisplay.from_predictions(y_val, lista_05, display_labels=['paga', 'incumple'],
                                                 cmap='Blues', colorbar=False)
matriz.ax_.set(xlabel='lo que predijo el modelo', ylabel='lo que realmente pasó',
               title='Umbral 0,5 · clientes de validación')
plt.show()
"""),
    code("""
accuracy_modelo = accuracy_score(y_val, lista_05)   # la diagonal sobre el total
display(Markdown(
    f'Accuracy: ({num(vn)} + {num(vp)}) / {num(len(y_val))} = **{num(100 * accuracy_modelo, 1)} %**, '
    f'contra **{num(100 * accuracy_baseline, 1)} %** del baseline. Casi no los separa, y sin embargo el '
    f'modelo detecta **{num(vp)}** incumplidores donde el baseline detectaba **{num(detectados_baseline)}**.'
))
"""),
    md("""
## Precision y recall: una columna y una fila

- **Precision**, sobre la columna «predijo incumple»: de las alarmas, qué parte era un impago de
  verdad. No se traduce como «precisión», que suena a accuracy.
- **Recall**, sobre la fila «realmente incumple»: de los que incumplieron, qué parte se detectó.

`precision_score` y `recall_score` reciben lo observado y lo predicho, y las calculan respecto de la
clase 1.

<!-- contrato: accion=precision-recall -->
"""),
    code("""
from sklearn.metrics import precision_score, recall_score  # las dos lecturas de la matriz

precision_05 = precision_score(y_val, lista_05)
recall_05 = recall_score(y_val, lista_05)
display(Markdown(
    f'- **precision** = VP / (VP + FP) = {num(vp)} / ({num(vp)} + {num(fp)}) = '
    f'**{num(100 * precision_05)} %**: de cada 100 alarmas, unas {num(100 * precision_05)} aciertan.\\n'
    f'- **recall** = VP / (VP + FN) = {num(vp)} / ({num(vp)} + {num(fn)}) = '
    f'**{num(100 * recall_05)} %**: de cada 100 que incumplen, se detectan unos {num(100 * recall_05)}.'
))
"""),
    md("""
### F1: precision y recall en una sola cifra

**F1** es su **media armónica**: solo es alto si las dos lo son. En conteos,
F1 = 2 × VP / (2 × VP + FP + FN), así que una falsa alarma y un impago que se escapa **pesan lo
mismo**. Sirve para comparar cuando los dos errores importan igual, no para elegir el umbral cuando
cuestan distinto. Lo calcula `f1_score`.

<!-- contrato: accion=f1 -->
"""),
    code("""
from sklearn.metrics import f1_score  # media armónica de precision y recall

f1_05 = f1_score(y_val, lista_05)
promedio_simple = (precision_05 + recall_05) / 2
display(Markdown(
    f'Con el umbral en 0,5: F1 = 2 × {num(vp)} / (2 × {num(vp)} + {num(fp)} + {num(fn)}) = '
    f'**{num(f1_05, 2)}**. El promedio simple de precision y recall daría {num(promedio_simple, 2)}: '
    f'F1 queda más cerca de la que falla, que es el recall.'
))
"""),
    md("""
## Mover el umbral: el barrido sobre validación

0,5 supone dos cosas a la vez: que los dos errores cuestan lo mismo y que la probabilidad es fiel.
Como no comprobamos ninguna, el umbral se **mide** sobre validación: una grilla de 0,05 a 0,90 y, para
cada umbral, sus dos errores, precision y recall. **El modelo no se reentrena**: solo cambia dónde se
corta.

<!-- contrato: accion=barrer-umbral -->
"""),
    code("""
GRILLA = np.round(np.arange(0.05, 0.91, 0.05), 2)   # de 0,05 a 0,90, de a 0,05

filas = []
for umbral in GRILLA:
    lista = np.where(prob_val >= umbral, 1, 0)   # las mismas probabilidades, otro umbral
    _, falsas, escapados, _ = confusion_matrix(y_val, lista, labels=[0, 1]).ravel()
    filas.append({'umbral': umbral, 'falsas_alarmas': falsas, 'impagos_escapados': escapados,
                  'precision': precision_score(y_val, lista), 'recall': recall_score(y_val, lista)})
barrido = pd.DataFrame(filas)
barrido.round(2)
"""),
    code("""
bajo, alto = barrido.iloc[0], barrido.iloc[-1]   # los dos extremos de la grilla
display(Markdown(
    f'Los extremos primero. Con **{num(bajo["umbral"], 2)}** se escapan **{num(bajo["impagos_escapados"])}** '
    f'impagos a cambio de **{num(bajo["falsas_alarmas"])}** falsas alarmas; con **{num(alto["umbral"], 2)}** '
    f'quedan **{num(alto["falsas_alarmas"])}** falsas alarmas, pero se escapan '
    f'**{num(alto["impagos_escapados"])}** de los {num(y_val.sum())} impagos. La decisión está entre los dos.'
))
"""),
    md("""
## El costo de cada error decide el umbral

Cuánto cuesta cada error lo decide la organización. Si acuerda que dejar pasar un impago cuesta
**cinco** llamadas de más:

> **costo total = 5 × impagos que se escapan + 1 × falsas alarmas**

Se trabaja con una **razón** y no con montos, porque una razón se acuerda en una reunión. Primero tres
umbrales; después, el más barato de la grilla.

<!-- contrato: accion=costo-cinco -->
"""),
    code("""
# un impago que se escapa vale 5 llamadas de más; una falsa alarma, 1
barrido['costo_5_a_1'] = 5 * barrido['impagos_escapados'] + barrido['falsas_alarmas']
display(barrido.loc[barrido['umbral'].isin([0.10, 0.15, 0.50]),
                    ['umbral', 'falsas_alarmas', 'impagos_escapados', 'costo_5_a_1']])

mas_barato = barrido.loc[barrido['costo_5_a_1'].idxmin()]
costo_1_a_1 = barrido['impagos_escapados'] + barrido['falsas_alarmas']   # los dos errores pesan igual
umbral_1_a_1 = barrido.loc[costo_1_a_1.idxmin(), 'umbral']
display(Markdown(
    f'En toda la grilla, el umbral más barato con razón 5 a 1 es **{num(mas_barato["umbral"], 2)}**, con '
    f'un costo de **{num(mas_barato["costo_5_a_1"])}**. Si los dos errores costaran lo mismo, sería '
    f'**{num(umbral_1_a_1, 2)}** en esta validación. No es una regla general: la grilla avanza de a 0,05 '
    f'y no comprobamos que la probabilidad sea fiel.'
))
"""),
    md("""
## Una cifra para todos los umbrales: ROC-AUC

La curva **ROC** marca, para cada umbral, la tasa de verdaderos positivos (el recall) contra la tasa
de falsos positivos (la parte de los que pagan que recibe una falsa alarma). **AUC** es el área bajo
ella: 0,5 es ordenar al azar y 1, ordenar perfecto. `RocCurveDisplay` la dibuja a partir de lo
observado y la probabilidad; los dos puntos son los umbrales 0,50 y 0,15 del barrido.
"""),
    code("""
from sklearn.metrics import RocCurveDisplay  # dibuja la curva ROC

curva = RocCurveDisplay.from_predictions(y_val, prob_val, name='logística', plot_chance_level=True)
for u in (0.50, 0.15):  # los dos umbrales de la clase, como puntos de la curva
    fila = barrido.loc[np.isclose(barrido['umbral'], u)].iloc[0]
    x, y_punto = fila['falsas_alarmas'] / (y_val == 0).sum(), fila['recall']
    curva.ax_.scatter(x, y_punto, s=80, color='darkgreen', zorder=3)
    curva.ax_.annotate(f'umbral {num(u, 2)}', (x, y_punto), xytext=(8, -14), textcoords='offset points')
curva.ax_.legend([curva.line_, curva.chance_level_], ['logística', 'al azar (área 0,5)'])
curva.ax_.set(xlabel='tasa de falsos positivos (falsas alarmas)',
              ylabel='tasa de verdaderos positivos (recall)', title='Curva ROC · clientes de validación')
plt.show()
"""),
    md("""
`roc_auc_score` recibe lo observado y la **probabilidad**, no la etiqueta, porque resume qué tan bien
**ordena** el modelo. La celda también lo lee como pares: cada cliente que incumple contra cada uno
que paga.

<!-- contrato: accion=roc-auc -->
"""),
    code("""
from sklearn.metrics import roc_auc_score  # resume el orden en todos los umbrales a la vez

auc_val = roc_auc_score(y_val, prob_val)
incumplen = prob_val[y_val.to_numpy() == 1]  # probabilidad de cada cliente que incumplió
pagan = prob_val[y_val.to_numpy() == 0]      # y de cada cliente que pagó
bien_ordenados = (incumplen[:, None] > pagan).mean()  # cada uno que incumple contra cada uno que paga
display(Markdown(
    f'Sobre validación, el ROC-AUC vale **{num(auc_val, 2)}**: de los '
    f'**{num(len(incumplen) * len(pagan))}** pares posibles, en **{num(100 * bien_ordenados)} de cada '
    f'100** el cliente que incumple recibe más probabilidad que el que paga. Mide el **orden**, no la '
    f'**lista**: el umbral se elige con la matriz y el costo.'
))
"""),
    md("""
# Tiempo 3 · Lo hacen ustedes

## La actividad: elegir un umbral y defenderlo

Cobranza y comercial no se ponen de acuerdo en cuánto cuesta cada error:

- **Cobranza:** detrás de un impago vienen el saldo expuesto y su recuperación. **Un impago que se
  escapa vale 4 llamadas de más.**
- **Comercial:** llamar a quien iba a pagar molesta al cliente y consume horas del equipo. **Un impago
  que se escapa vale 1,5 llamadas de más.**

1. **Elijan una posición** y escriban su razón en la celda `razon`.
2. **Corran el costo de cada umbral** sobre validación y anoten el más barato.
3. **Abran prueba una sola vez** con ese umbral y anoten cuánto ahorra frente a 0,5.
4. **Defiendan su lista en tres líneas:** a quién llaman, cuántos impagos se les escapan y quién paga
   ese costo.

**Criterio de éxito:** las tres líneas nombran los dos errores y qué le pasa a cada uno con el umbral
elegido.

**Evidencia observable:** la tabla del barrido, el umbral con su posición y las tres líneas; el
**feedback formativo** llega en la puesta en común.

### Paso 1 · Elegir una posición

La celda corre con cobranza. Para comercial, cambien el `4` por `1.5` —con punto, porque es código— y
vuelvan a correr desde ahí. Si quieren hacer la actividad sin ver antes el resultado de cobranza, borren
primero los resultados (en Colab, Editar → Borrar todos los resultados) y corran desde el inicio.

<!-- contrato: accion=elegir-posicion -->
"""),
    code("""
# si eligen comercial, cambien el 4 por 1.5 y vuelvan a correr desde esta celda
razon = 4  # cobranza: 4 · comercial: 1.5
"""),
    md("""
### Paso 2 · El costo de cada umbral, sobre validación

La cuenta de la razón 5 a 1, con el peso de su posición. Prueba sigue cerrada.

<!-- contrato: accion=elegir-umbral -->
"""),
    code("""
barrido['costo'] = razon * barrido['impagos_escapados'] + barrido['falsas_alarmas']
umbral_elegido = barrido.loc[barrido['costo'].idxmin(), 'umbral']

dec = 0 if float(razon).is_integer() else 1   # 4 se escribe «4»; 1.5, «1,5»
display(Markdown(f'Con razón **{num(razon, dec)} a 1**, el umbral más barato sobre validación es '
                 f'**{num(umbral_elegido, 2)}**.'))
display(barrido[['umbral', 'falsas_alarmas', 'impagos_escapados', 'precision', 'recall', 'costo']].round(2))
"""),
    md("""
### Paso 3 · Abrir prueba, una sola vez

La razón la puso su posición y el umbral lo eligió validación: ya no queda nada por decidir. Es la
**única celda que lee prueba**, y el modelo no se reajusta antes, porque el umbral se eligió para sus
probabilidades.

<!-- contrato: accion=abrir-la-prueba -->
"""),
    code("""
prob_test = modelo.predict_proba(X_test)[:, 1]   # prueba se abre acá, y solo acá
_, fp_elegido, fn_elegido, _ = confusion_matrix(
    y_test, np.where(prob_test >= umbral_elegido, 1, 0), labels=[0, 1]).ravel()
_, fp_defecto, fn_defecto, _ = confusion_matrix(
    y_test, np.where(prob_test >= UMBRAL_DEFECTO, 1, 0), labels=[0, 1]).ravel()

costo_elegido = razon * fn_elegido + fp_elegido
costo_defecto = razon * fn_defecto + fp_defecto
ahorro = (costo_defecto - costo_elegido) / costo_defecto
"""),
    code("""
display(Markdown(
    f'Sobre los **{num(len(prob_test))}** clientes de prueba, con razón {num(razon, dec)} a 1:\\n\\n'
    f'- con su umbral, **{num(umbral_elegido, 2)}**: {num(fp_elegido)} llamadas de más y '
    f'{num(fn_elegido)} impagos que se escapan, con un costo de **{num(costo_elegido, dec)}**;\\n'
    f'- con **{num(UMBRAL_DEFECTO, 2)}**: {num(fp_defecto)} llamadas de más y {num(fn_defecto)} '
    f'impagos que se escapan, con un costo de **{num(costo_defecto, dec)}**.\\n\\n'
    f'Su umbral ahorra el **{num(100 * ahorro)} %** del costo frente a 0,5.'
))
"""),
    md("""
### Paso 4 · Su defensa, en tres líneas

El umbral **reparte el error entre personas**: bajarlo detecta más impagos y señala a más clientes
que iban a pagar. Por eso la decisión es de la organización, no del algoritmo. Con las cifras de los
pasos 2 y 3:

1. A quién llaman:
2. Cuántos impagos se les escapan:
3. Quién paga ese costo:
"""),
    md("""
# Cierre

## A quién llamar, y cómo defender la lista

Cobranza llama a los clientes cuya probabilidad alcanza el umbral que **minimiza el costo total** con
la razón que acuerda la organización. La accuracy no defiende esa lista; la matriz y el costo, sí. El
umbral se elige **sobre validación** y el ahorro se informa **con prueba, abierta una sola vez**. Si
cambia la razón, cambia la lista; el modelo es el mismo.

Ese umbral lo eligió **una** partición de validación, y con otra podría ser otro. Cómo elegir sin
depender de una sola partición es la pregunta de la sesión 8.
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
    """Cuenta las llamadas a `nombre` y su cantidad de argumentos, por AST.

    LA CANTIDAD DE ARGUMENTOS TAMBIÉN SE DERIVA, no se escribe a mano. Falla cerrado si las llamadas
    no coinciden entre sí, porque entonces un único número no las describe.
    """
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


def celdas_que_leen(codigo: list[dict], nombre: str) -> int:
    """Celdas de código que LEEN `nombre`, con la misma regla de `scripts/check_notebook.py`."""
    return sum(
        any(isinstance(n, ast.Name) and n.id == nombre and isinstance(n.ctx, ast.Load)
            for n in ast.walk(_arbol("".join(c["source"]))))
        for c in codigo
    )


def contrato(celdas: list[dict]) -> dict:
    """Deriva `metadata.curso_contrato` de la lista FINAL de celdas.

    Se calcula, no se escribe a mano: agregar, quitar o mover una celda lo actualiza solo.
    """
    codigo = [c for c in celdas if c["cell_type"] == "code"]
    markdown = sum(1 for c in celdas if c["cell_type"] == "markdown")
    assert markdown >= len(codigo), (
        f"{markdown} celdas markdown contra {len(codigo)} de código: el cuaderno es material de "
        "estudio y la explicación va en Markdown")
    fuente = "\n".join("".join(c["source"]) for c in codigo)
    lineas_por_celda = [
        len([l for l in "".join(c["source"]).splitlines()
             if l.strip() and not l.strip().startswith("#")])
        for c in codigo
    ]
    # Prueba se abre UNA vez. Se mide, no se declara: si otra celda la lee, falla acá.
    lectura_prueba = {v: celdas_que_leen(codigo, v) for v in ("X_test", "y_test")}
    assert lectura_prueba == {"X_test": 1, "y_test": 1}, (
        f"prueba la lee más de una celda (o ninguna): {lectura_prueba}")
    return {
        "forma": {
            "markdown": markdown,
            "codigo": len(codigo),
            "llamadas": [llamada(fuente, "train_test_split"), llamada(fuente, "confusion_matrix")],
        },
        "acciones": [
            {"marcador": "fijar-base", "asignacion": "datos"},
            {"marcador": "declarar-positiva", "asignacion": "y"},
            {"marcador": "partir-en-tres", "codigo": "train_test_split"},
            {"marcador": "el-baseline", "codigo": "DummyClassifier"},
            {"marcador": "el-modelo", "codigo": "LogisticRegression"},
            {"marcador": "la-probabilidad", "codigo": "predict_proba"},
            {"marcador": "odds-ratio", "asignacion": "odds_ratio"},
            {"marcador": "la-matriz", "codigo": "confusion_matrix"},
            {"marcador": "precision-recall", "codigo": "precision_score"},
            {"marcador": "f1", "codigo": "f1_score"},
            {"marcador": "barrer-umbral", "asignacion": "barrido"},
            {"marcador": "costo-cinco", "asignacion": "costo_5_a_1"},
            {"marcador": "roc-auc", "codigo": "roc_auc_score"},
            {"marcador": "elegir-posicion", "asignacion": "razon"},
            {"marcador": "elegir-umbral", "asignacion": "umbral_elegido"},
            {"marcador": "abrir-la-prueba", "asignacion": "ahorro"},
        ],
        # Es una MEDIANA, no un máximo por celda: deja pasar el setup largo y sigue exigiendo que el
        # cuaderno esté partido en unidades chicas. Se fija al valor real medido, con un margen de 1.
        "max_mediana_lineas_codigo": int(statistics.median(lineas_por_celda)) + 1,
        "max_celdas_lectura": lectura_prueba,
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
    print(f"  contrato: mediana de líneas {c['max_mediana_lineas_codigo'] - 1}"
          f" · llamadas {c['forma']['llamadas']} · lectura de prueba {c['max_celdas_lectura']}")


if __name__ == "__main__":
    main()
