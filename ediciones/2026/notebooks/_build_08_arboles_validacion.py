#!/usr/bin/env python3
"""Construye notebooks/08-arboles-validacion.ipynb.

POR QUÉ EXISTE UN GENERADOR: el `.ipynb` no se edita a mano. Si hay que corregir algo, se corrige acá y
se vuelve a ejecutar con `make nb-build C=08`.

ESTRUCTURA: los tres tiempos de la `practica` del temario, más el cierre. (1) El docente lee un árbol de
dos preguntas con `plot_tree`, muestra que el árbol sin límite memoriza y MIDE la inestabilidad: un
árbol y un bosque reentrenados con veinte particiones, mirando por separado lo que predicen para un
grupo de referencia y su cifra de validación. (2) Entre todos: la validación cruzada de cinco pliegues
sobre la logística de la sesión 7, la curva en U con `validation_curve` y la búsqueda del árbol con
`GridSearchCV`, con `cv_results_` completo. (3) La actividad: contar el costo de la grilla del bosque,
correrla, comparar tres modelos con los mismos pliegues, elegir uno, abrir prueba UNA vez y defenderlo.

LA MEDICIÓN ES LA DEL GENERADOR DE FIGURAS (`assets/ilustraciones/_build_c08_figuras.py`): mismo
archivo, mismas predictoras en el mismo orden, mismas particiones, mismo grupo de referencia, mismos
pliegues y mismas grillas. Por eso las cifras del cuaderno coinciden con las del deck sin copiarse: se
RECALCULAN en cada celda y se muestran con Markdown dinámico por f-string. La curva en U es la única
que difiere a propósito: el deck la muestra con una validación, antes de enseñar los pliegues; el
cuaderno, que llega después de la teoría completa, la calcula con validación cruzada.

PRUEBA SE ABRE UNA SOLA VEZ, en el paso 4 de la actividad y para el modelo que el grupo eligió. El
tamaño de prueba se informa como lo que quedó fuera de `X_trainval`, sin leerla. El contrato lo protege
con `max_celdas_lectura`, derivado por AST y comprobado aquí con un `assert`.

LA COMPUERTA DE PRUEBA (auditoría Codex F5E). El cuaderno se publica con `elegido = None`: una versión
anterior traía el bosque elegido y publicaba su cifra de prueba antes de que nadie decidiera. La celda que
abre prueba registra el modelo en `pruebas_abiertas`, que vive en la celda de la partición para que volver
a correr la elección no lo borre, y no abre prueba para un segundo modelo. Sus tres caminos se ejecutan
aislados:

    make nb C=08 SCENARIO=ediciones/2026/notebooks/_escenarios/08-saltar.py
    make nb C=08 SCENARIO=ediciones/2026/notebooks/_escenarios/08-completar.py
    make nb C=08 SCENARIO=ediciones/2026/notebooks/_escenarios/08-volver.py

UNA PREPARACIÓN NUEVA POR PIPELINE: la preparación se define una vez y cada pipeline recibe
`clone(...)` de ella, nunca el mismo objeto ni una función fábrica (ver `diseno-clase`, notebooks). Los
árboles y el bosque van sin escalar: una pregunta «¿mayor que 2?» no cambia con la escala, y así
`plot_tree` muestra los valores en unidades del archivo. La logística escala, como en la sesión 7.

PYTHON BÁSICO (docente, 2026-09-09): sin listas por comprensión ni `lambda`. Los ciclos son `for` con
`append`.

FRONTERAS, heredadas del `alcance_no` y verificadas al escribir:
  · Gradient boosting SOLO como cuarto candidato de la actividad (docente, 2026-10-16), con
    `HistGradientBoostingClassifier` y una grilla dada; su mecanismo no se desarrolla.
  · Ni búsqueda aleatoria implementada, ni búsqueda guiada.
  · Ni importancia de variables del bosque: es la sesión de interpretabilidad.
  · Ni medidas de impureza, ni error out-of-bag, ni nested cross-validation.
  · `TimeSeriesSplit` y `GroupKFold` no se usan: el archivo no tiene orden temporal ni clientes repetidos.

Uso:
    make nb-build C=08   (regenera, ejecuta en sandbox y deja el ejecutado publicado)
    make nb-check C=08   (verifica el contrato)
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
SALIDA = AQUI / "08-arboles-validacion.ipynb"
ILUSTRACIONES = REPO_ROOT / "assets" / "ilustraciones"
PREP_PUBLICO = ("https://github.com/sebastiancontz/ust-introduccion-machine-learning-colab/blob/main/"
                "ediciones/2026/datasets/_prep_riesgo_impago.py")


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
<p>Semana 08: Árboles, ensambles y validación cruzada</p>
</div>
</div>

# 08 · Árboles, ensambles y validación cruzada

> **La pregunta de hoy:** si cobranza cambia la logística por un bosque, ¿cómo sabemos que es mejor y
> no la suerte de una partición?

Tres tiempos: **yo lo hago** (el árbol y su inestabilidad, medida), **lo hacemos juntos** (validación
cruzada, curva en U y búsqueda) y **lo hacen ustedes** (elegir entre cuatro modelos y defenderlo).
"""),
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas; la celda queda para correr fuera de Colab.
"""),
    code("""
%%capture
!pip install -q pandas numpy scikit-learn matplotlib
"""),
    md("""
## Cargar datos

La cartera de **tarjetas de crédito** de la sesión 7, `riesgo_impago.csv`, leída con **pandas**. La
celda define también `num`, que escribe las cifras con punto de miles y coma decimal.

<!-- contrato: accion=fijar-base -->
"""),
    code("""
import os                # revisa si el archivo está en una carpeta local
import pandas as pd      # la librería de tablas: lee el archivo y lo deja como DataFrame
from IPython.display import Markdown, display  # muestra frases con las cifras ya calculadas

def num(x, dec=0):  # miles con punto, decimales con coma y el menos tipográfico
    texto = f'{x:,.{dec}f}'.replace(',', '@').replace('.', ',').replace('@', '.')
    return texto.replace('-', '−')

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

datos = pd.read_csv(BASE + 'riesgo_impago.csv')
print('registros y columnas:', datos.shape)
"""),
    md("""
## El encuadre, el mismo de la sesión 7

- **Unidad de observación:** un registro es un **cliente observado en un mes**.
- **Objetivo:** si el cliente **incumple el pago del mes siguiente** (`incumplio_pago`): 1 = incumple,
  la **clase positiva**; 0 = paga.
- **Métrica:** **ROC-AUC**, la que la sesión 7 dejó medida para la logística: resume qué tan bien
  ordena el modelo a los clientes, de 0,5 (al azar) a 1 (perfecto).

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
print(len(COLUMNAS_NUM), 'numéricas y', len(COLUMNAS_CAT), 'categórica')
"""),
    md("""
## La partición: la de la sesión 7, con prueba apartada

Misma semilla y mismo `stratify=y`. Prueba se aparta y no se vuelve a tocar hasta la actividad.

- `X_trainval`: **entrenamiento y validación juntos**. Es donde rotan los pliegues de la validación
  cruzada.
- `X_train` y `X_val`: la partición de la sesión 7, que usamos solo para las demostraciones de una
  sola partición, antes de llegar a los pliegues.

<!-- contrato: accion=partir -->
"""),
    code("""
from sklearn.model_selection import train_test_split  # sortea la partición

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)
pruebas_abiertas = []   # para qué modelo se abre prueba: se llena una sola vez, en la actividad

display(Markdown(
    f'`X_trainval`: **{num(len(X_trainval))}** clientes (entrenamiento {num(len(X_train))} + validación '
    f'{num(len(X_val))}) · prueba: **{num(len(datos) - len(X_trainval))}**, guardada hasta la actividad.'
))
"""),
    md("""
# Tiempo 1 · Yo lo hago

## Un árbol de dos preguntas: `DecisionTreeClassifier`

Un **árbol de decisión** predice con preguntas de sí o no encadenadas, cada una sobre una columna.
`DecisionTreeClassifier` aprende de los datos qué preguntar y con qué valor; `max_depth=2` lo limita a
dos preguntas encadenadas.

La preparación va **sin escalar** (`'passthrough'`): una pregunta como «¿2 o más meses de mora?» da lo
mismo en cualquier escala, y así el árbol muestra los valores en unidades del archivo. Se define una vez,
y cada pipeline recibe una copia nueva con `clone`, para que entrenar uno no cambie a otro.

<!-- contrato: accion=arbol-dos -->
"""),
    code("""
from sklearn.base import clone                    # copia sin entrenar de una preparación
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier   # el árbol de decisión

preparar_arbol = ColumnTransformer([
    ('num', 'passthrough', COLUMNAS_NUM),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUMNAS_CAT),
], verbose_feature_names_out=False)  # nombres de columna sin prefijo, para leer el árbol

arbol_2 = Pipeline([('preparar', clone(preparar_arbol)),
                    ('modelo', DecisionTreeClassifier(max_depth=2, random_state=42))])
arbol_2.fit(X_train, y_train);  # el punto y coma evita imprimir el pipeline entero
"""),
    md("""
### El árbol, dibujado con `plot_tree`

`plot_tree` recibe el árbol ajustado —el paso `modelo` del pipeline— y los nombres de las columnas que
salen de la preparación. Cada caja muestra la pregunta, cuántos clientes llegan (`samples`) y cuántos
de cada clase (`value`); el color se intensifica hacia la clase que predice. Se lee de arriba abajo:
`meses_mora <= 1.5` es «¿1 mes de mora o menos?», y si la respuesta es sí se va a la izquierda. Cada
hoja predice la **proporción que incumplió** en su grupo de entrenamiento; debajo, tres clientes de
validación la recorren.
"""),
    code("""
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree   # dibuja un árbol ajustado

nombres = arbol_2.named_steps['preparar'].get_feature_names_out()
plt.figure(figsize=(12, 5))
plot_tree(arbol_2.named_steps['modelo'], feature_names=list(nombres),
          class_names=['paga', 'incumple'], filled=True, impurity=False, fontsize=10)
plt.title(f'Árbol de profundidad 2 · {num(len(X_train))} clientes de entrenamiento')
plt.show()
"""),
    code("""
tres = ['R-4608', 'R-5805', 'R-2647']   # un cliente por hoja distinta
en_validacion = datos.loc[X_val.index]
marcados = en_validacion['id_cliente'].isin(tres).to_numpy()
tabla = en_validacion.loc[marcados, ['id_cliente', 'meses_mora', 'meses_mora_previo',
                                     'meses_mora_acumulados', 'incumplio_pago']]
tabla['probabilidad del árbol'] = arbol_2.predict_proba(X_val[marcados])[:, 1].round(2)
tabla
"""),
    md("""
## Sin límite, el árbol memoriza

Sin `max_depth`, el árbol sigue preguntando hasta que casi cada hoja tiene un solo tipo de cliente.
`get_n_leaves` cuenta sus hojas, y `roc_auc_score`, el de la sesión 7, lo mide en entrenamiento y en
validación.

<!-- contrato: accion=arbol-libre -->
"""),
    code("""
from sklearn.metrics import roc_auc_score   # recibe lo observado y la probabilidad

arbol_libre = Pipeline([('preparar', clone(preparar_arbol)),
                        ('modelo', DecisionTreeClassifier(random_state=42))])
arbol_libre.fit(X_train, y_train)
auc_train = roc_auc_score(y_train, arbol_libre.predict_proba(X_train)[:, 1])
auc_val = roc_auc_score(y_val, arbol_libre.predict_proba(X_val)[:, 1])
display(Markdown(
    f'**{num(arbol_libre.named_steps["modelo"].get_n_leaves())} hojas**. ROC-AUC: **{num(auc_train, 2)}** '
    f'en entrenamiento y **{num(auc_val, 2)}** en validación. Ordena perfecto a quienes vio y apenas '
    f'mejor que el azar a quienes no.'
))
"""),
    md("""
## La inestabilidad, medida: árbol contra bosque

Un **random forest** (bosque aleatorio) entrena muchos árboles, cada uno con una **remuestra** distinta
de los clientes de entrenamiento y eligiendo cada pregunta entre **algunas columnas sorteadas**, y
promedia las probabilidades que entregan. `RandomForestClassifier` lo hace con 100 árboles por defecto; `n_jobs=-1` usa todos los
núcleos del computador.

El experimento separa las **dos variaciones** de la clase:

1. Se apartan **600 clientes de referencia** de `X_trainval`. No entran a ningún entrenamiento ni
   validación de este experimento: solo se mira qué predice cada modelo para ellos.
2. Con los demás se arman **20 particiones** distintas, y en cada una se entrena un árbol sin límite y
   un bosque. El ciclo tarda unos segundos y guarda, por modelo, lo que predice para los clientes de
   referencia y su ROC-AUC de validación.

Después se miran las dos variaciones por separado. **La del modelo:** un cliente de referencia
**cambia de clase** si, con el umbral de 0,5, queda de un lado en alguna partición y del otro en otra;
NumPy lo cuenta con `any` y `all` sobre la tabla de 20 × 600 predicciones. Después, la de la
evaluación.

<!-- contrato: accion=grupo-referencia -->
<!-- contrato: accion=veinte-particiones -->
"""),
    code("""
from sklearn.ensemble import RandomForestClassifier   # el bosque aleatorio

X_resto, X_ref, y_resto, y_ref = train_test_split(   # los 600 de referencia quedan fuera del ciclo
    X_trainval, y_trainval, test_size=600, random_state=42, stratify=y_trainval)
pred_arbol, pred_bosque, auc_arbol, auc_bosque = [], [], [], []
for semilla in range(20):
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        X_resto, y_resto, test_size=0.25, random_state=semilla, stratify=y_resto)
    arbol = Pipeline([('preparar', clone(preparar_arbol)), ('modelo', DecisionTreeClassifier(random_state=42))])
    bosque = Pipeline([('preparar', clone(preparar_arbol)),
                       ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))])
    arbol.fit(X_train_s, y_train_s)
    bosque.fit(X_train_s, y_train_s)
    pred_arbol.append(arbol.predict_proba(X_ref)[:, 1])
    pred_bosque.append(bosque.predict_proba(X_ref)[:, 1])
    auc_arbol.append(roc_auc_score(y_val_s, arbol.predict_proba(X_val_s)[:, 1]))
    auc_bosque.append(roc_auc_score(y_val_s, bosque.predict_proba(X_val_s)[:, 1]))
"""),
    code("""
import numpy as np

def cambian_de_clase(predicciones):
    sobre_umbral = np.array(predicciones) >= 0.5      # 20 particiones × 600 clientes
    alguna, todas = sobre_umbral.any(axis=0), sobre_umbral.all(axis=0)
    return (alguna & ~todas).mean()                   # alguna vez sí y alguna vez no

cambia_arbol, cambia_bosque = cambian_de_clase(pred_arbol), cambian_de_clase(pred_bosque)
display(Markdown(
    f'Cambian de clase según la partición: **{num(100 * cambia_arbol)} %** de los clientes con el árbol '
    f'sin límite y **{num(100 * cambia_bosque)} %** con el bosque.'
))
"""),
    md("""
**La de la evaluación:** cuánto se mueve la cifra de validación de cada modelo entre las mismas 20
particiones.
"""),
    code("""
cifras = pd.DataFrame({'árbol sin límite': auc_arbol, 'bosque': auc_bosque})
resumen = pd.DataFrame({'media': cifras.mean(), 'mínimo': cifras.min(), 'máximo': cifras.max()})
resumen['rango'] = resumen['máximo'] - resumen['mínimo']
display(resumen.round(2))
display(Markdown(
    f'El bosque está más arriba, pero su cifra se mueve casi lo mismo: un rango de '
    f'**{num(resumen.loc["bosque", "rango"], 2)}** contra **{num(resumen.loc["árbol sin límite", "rango"], 2)}**.'
))
"""),
    md("""
### Dos variaciones, dos remedios

""" + figura_base64(
        "c08-dos-variaciones.svg",
        "Dos paneles. A la izquierda, el porcentaje de 600 clientes de referencia que cambian de clase "
        "según la partición: 79 por ciento con el árbol sin límite y 17 por ciento con el bosque. A la "
        "derecha, el ROC-AUC de validación en 20 particiones: el árbol con un rango de 0,08 y el bosque "
        "con un rango de 0,07.") + """

| Qué varía | Cómo se ve | Qué lo reduce |
|:--|:--|:--|
| **El modelo** | con otra muestra, otra predicción para el mismo cliente | el **bosque** |
| **La evaluación** | con otra partición, otra cifra para el mismo modelo | la **validación cruzada** |

El bosque no vuelve confiable la cifra de una partición. Para eso, el tiempo 2.
"""),
    md("""
# Tiempo 2 · Lo hacemos juntos

## La validación cruzada: `StratifiedKFold` y `cross_val_score`

`StratifiedKFold` reparte `X_trainval` en **5 pliegues** con la misma proporción de impagos cada uno;
`shuffle=True` mezcla antes de repartir. `cross_val_score` recibe el pipeline, los datos, los
pliegues y la métrica: en cada ronda entrena con cuatro pliegues, valida con el quinto, y devuelve las
cinco cifras. Calcula de nuevo la preparación en cada ronda, así que no hay fuga.

""" + figura_base64(
        "c08-pliegues.svg",
        "Esquema de validación cruzada de 5 pliegues sobre 4.800 clientes: cinco rondas, en cada una un "
        "pliegue distinto valida y los otros cuatro entrenan. A la derecha, el ROC-AUC de la regresión "
        "logística en cada ronda y un recuadro con la media y la desviación.") + """

Empezamos por la **logística de la sesión 7**, con su mismo pipeline. La desviación entre pliegues es
**orientativa**: los cinco entrenamientos comparten casi todos sus clientes, así que sirve para comparar
modelos con los **mismos** pliegues, no para demostrar que difieren.

<!-- contrato: accion=validar-cruzado -->
"""),
    code("""
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logistica = Pipeline([
    ('preparar', ColumnTransformer([('num', StandardScaler(), COLUMNAS_NUM),
                                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), COLUMNAS_CAT)])),
    ('modelo', LogisticRegression(max_iter=2000)),
])
cv_logistica = cross_val_score(logistica, X_trainval, y_trainval, cv=pliegues, scoring='roc_auc')
display(Markdown(
    f'Cinco cifras, de {num(cv_logistica.min(), 3)} a {num(cv_logistica.max(), 3)}. Se informan la '
    f'**media, {num(cv_logistica.mean(), 3)}**, y la '
    f'**desviación, {num(cv_logistica.std(), 3)}**, que dice cuánto se movió la cifra entre pliegues.'
))
"""),
    md("""
## La curva en U, con `validation_curve`

`validation_curve` hace la validación cruzada para cada valor de **un** hiperparámetro y devuelve dos
tablas: el puntaje de entrenamiento y el de validación, una fila por valor y una columna por pliegue.
`param_name` dice qué hiperparámetro mover —el paso `modelo`, dos guiones bajos y su nombre— y
`param_range`, qué valores probar; `None` es «sin límite».

<!-- contrato: accion=curva-u -->
"""),
    code("""
from sklearn.model_selection import validation_curve

PROFUNDIDADES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
arbol = Pipeline([('preparar', clone(preparar_arbol)), ('modelo', DecisionTreeClassifier(random_state=42))])
puntaje_train, puntaje_val = validation_curve(
    arbol, X_trainval, y_trainval, param_name='modelo__max_depth', param_range=PROFUNDIDADES,
    cv=pliegues, scoring='roc_auc')
curva = pd.DataFrame({'entrenamiento': puntaje_train.mean(axis=1), 'validación': puntaje_val.mean(axis=1)},
                     index=PROFUNDIDADES)
"""),
    code("""
rotulos = []
for p in PROFUNDIDADES:
    rotulos.append('sin límite' if p is None else str(p))
ax = curva.plot(marker='o', figsize=(9, 4), color=['#D55E00', '#0072B2'], xticks=range(len(rotulos)),
                use_index=False, title=f'Curva en U · {num(len(X_trainval))} clientes, media de 5 pliegues')
ax.set_xticklabels(rotulos)
ax.set(xlabel='profundidad máxima del árbol', ylabel='ROC-AUC (más alto es mejor)')
plt.show()

orden = np.argsort(curva['validación'].to_numpy())   # de la peor a la mejor profundidad
mejor, segunda = rotulos[orden[-1]], rotulos[orden[-2]]  # el rótulo: None se vuelve «sin límite»
display(Markdown(
    f'Entrenamiento sube siempre, hasta **{num(curva["entrenamiento"].iloc[-1], 2)}** sin límite. '
    f'Validación llega a su punto más alto con profundidad **{mejor}** ({num(curva["validación"].max(), 3)}), '
    f'casi empatada con **{segunda}** ({num(curva["validación"].iloc[orden[-2]], 3)}), y cae a '
    f'**{num(curva["validación"].iloc[-1], 3)}** sin límite. Más alto es mejor: la U se ve al revés.'
))
"""),
    md("""
## La búsqueda: `GridSearchCV`

`GridSearchCV` automatiza el procedimiento de la sesión 6: recibe el pipeline y una **grilla**, prueba
**cada combinación** con los pliegues y la métrica, y al final **vuelve a entrenar la mejor** con todo
`X_trainval`. Dos hiperparámetros del árbol: la profundidad y el mínimo de clientes por hoja
(`min_samples_leaf`), que prohíbe hojas más chicas.

<!-- contrato: accion=buscar-arbol -->
"""),
    code("""
from sklearn.model_selection import GridSearchCV

grilla_arbol = {'modelo__max_depth': [3, 5, 8, None], 'modelo__min_samples_leaf': [1, 20, 100]}
busqueda_arbol = GridSearchCV(arbol, grilla_arbol, cv=pliegues, scoring='roc_auc')
busqueda_arbol.fit(X_trainval, y_trainval)
combinaciones = len(grilla_arbol['modelo__max_depth']) * len(grilla_arbol['modelo__min_samples_leaf'])
display(Markdown(f'{combinaciones} combinaciones × 5 pliegues = **{combinaciones * 5} entrenamientos**, '
                 f'y uno más para volver a entrenar la mejor.'))
"""),
    md("""
### Todos los intentos: `cv_results_`

`cv_results_` guarda **cada intento**, no solo el ganador: sus hiperparámetros, la media y la desviación
de los pliegues, y su lugar. Se ordena por `rank_test_score`, el lugar, con 1 para el mejor.
"""),
    code("""
intentos = pd.DataFrame(busqueda_arbol.cv_results_)
intentos = intentos[['param_modelo__max_depth', 'param_modelo__min_samples_leaf',
                     'mean_test_score', 'std_test_score', 'rank_test_score']]
intentos = intentos.sort_values('rank_test_score')
display(intentos.round(3))

ganador = intentos.iloc[0]
banda = ganador['mean_test_score'] - ganador['std_test_score']
empatados = (intentos['mean_test_score'] >= banda).sum()
display(Markdown(
    f'Gana **{num(ganador["mean_test_score"], 3)} ± {num(ganador["std_test_score"], 3)}**, pero '
    f'**{empatados}** intentos, contando al ganador, quedan a menos de una desviación de él: con estos '
    f'pliegues, no se distinguen. Entre los empatados conviene el más simple; es una heurística, no una demostración.'
))
"""),
    md("""
# Tiempo 3 · Lo hacen ustedes

## La actividad: ¿bosque, boosting, árbol o logística?

Cobranza evalúa cambiar la logística por un modelo de árboles. Tienen cuatro candidatos: la
**logística** de la sesión 7, el **árbol** que eligió la búsqueda, y un **bosque** y un **boosting** que
van a ajustar ustedes.

1. **Cuenten** cuántos entrenamientos costarán las dos grillas, **antes** de correrlas.
2. **Corran las búsquedas** y lean `cv_results_` completo.
3. **Comparen los cuatro con los mismos pliegues**: media, desviación y en cuántos pliegues le gana cada
   uno a la logística.
4. **Elijan uno, abran prueba una sola vez** para ese modelo y defiendan su elección en tres líneas.

**Criterio de éxito:** la defensa cita la media **y** la desviación de los candidatos, dice si la diferencia
es mayor que la desviación, y pesa lo que se pierde al no poder leer el modelo como reglas.

**Evidencia observable:** la cuenta de entrenamientos, la tabla de comparación y las tres líneas; el
**feedback formativo** llega en la puesta en común.

### Paso 1 · Contar antes de correr

Dos hiperparámetros del bosque: cuántas columnas puede sortear en cada pregunta (`max_features`:
`'sqrt'` es la raíz cuadrada del total; `0.5`, la mitad) y el mínimo de clientes por hoja. Dos del
boosting: cuánto pesa la corrección de cada árbol nuevo (`learning_rate`) y la profundidad de cada árbol. Antes
de ejecutar la celda, hagan la cuenta a mano.

<!-- contrato: accion=contar-ajustes -->
"""),
    code("""
grilla_bosque = {'modelo__max_features': ['sqrt', 0.5], 'modelo__min_samples_leaf': [1, 5, 20, 50]}
combinaciones_bosque = len(grilla_bosque['modelo__max_features']) * len(grilla_bosque['modelo__min_samples_leaf'])
ajustes_bosque = combinaciones_bosque * 5
grilla_boosting = {'modelo__learning_rate': [0.03, 0.1], 'modelo__max_depth': [3, None]}
combinaciones_boosting = len(grilla_boosting['modelo__learning_rate']) * len(grilla_boosting['modelo__max_depth'])
display(Markdown(f'Bosque: {combinaciones_bosque} combinaciones × 5 pliegues = **{ajustes_bosque} entrenamientos**. '
                 f'Boosting: {combinaciones_boosting} × 5 = **{combinaciones_boosting * 5} entrenamientos**. '
                 f'Cada búsqueda suma uno para volver a entrenar a su ganador.'))
"""),
    md("""
### Paso 2 · La búsqueda del bosque

Tarda un poco más que la del árbol: cada ajuste son 100 árboles.

<!-- contrato: accion=buscar-bosque -->
"""),
    code("""
bosque = Pipeline([('preparar', clone(preparar_arbol)),
                   ('modelo', RandomForestClassifier(random_state=42, n_jobs=-1))])
busqueda_bosque = GridSearchCV(bosque, grilla_bosque, cv=pliegues, scoring='roc_auc')
busqueda_bosque.fit(X_trainval, y_trainval)

intentos_bosque = pd.DataFrame(busqueda_bosque.cv_results_)
intentos_bosque = intentos_bosque[['param_modelo__max_features', 'param_modelo__min_samples_leaf',
                                   'mean_test_score', 'std_test_score', 'rank_test_score']]
intentos_bosque.sort_values('rank_test_score').round(3)
"""),
    md("""
### Paso 2b · La búsqueda del boosting: `HistGradientBoostingClassifier`

El **gradient boosting** también combina árboles, pero no en paralelo como el bosque: los entrena **en
secuencia**, y cada árbol corrige lo que los anteriores todavía predicen mal. `HistGradientBoostingClassifier`
es la versión de scikit-learn: agrupa los valores de cada columna en intervalos antes de buscar las
preguntas, lo que la hace mucho más rápida cuando hay muchos datos. Su mecanismo no se desarrolla;
acá entra como un candidato más, con su grilla. Una diferencia con el bosque que sí conviene saber: el
boosting **puede sobreajustar** si se le agregan demasiados árboles.

""" + figura_base64(
        "c08-boosting-secuencia.svg",
        "Probabilidad de incumplir de dos clientes de validación a medida que el boosting suma árboles, de "
        "0 a 100. Los dos parten en 0,23, la proporción de impago de entrenamiento; la de R-3391, que "
        "incumplió, sube de a poco hasta 0,66, y la de R-5991, que pagó, baja hasta 0,09.") + """

<!-- contrato: accion=buscar-boosting -->
"""),
    code("""
from sklearn.ensemble import HistGradientBoostingClassifier   # árboles en secuencia

boosting = Pipeline([('preparar', clone(preparar_arbol)),
                     ('modelo', HistGradientBoostingClassifier(random_state=42))])
busqueda_boosting = GridSearchCV(boosting, grilla_boosting, cv=pliegues, scoring='roc_auc')
busqueda_boosting.fit(X_trainval, y_trainval)

intentos_boosting = pd.DataFrame(busqueda_boosting.cv_results_)
intentos_boosting = intentos_boosting[['param_modelo__learning_rate', 'param_modelo__max_depth',
                                       'mean_test_score', 'std_test_score', 'rank_test_score']]
intentos_boosting.sort_values('rank_test_score').round(3)
"""),
    md("""
### Paso 3 · Los cuatro, con los mismos pliegues

Las cinco cifras del ganador de cada búsqueda están en `cv_results_`, en las columnas `split0_test_score`
a `split4_test_score`, fila `best_index_`. Se suma el **bosque por defecto**, sin ajustar, para ver qué
aporta la búsqueda.

<!-- contrato: accion=comparar -->
"""),
    code("""
def cifras_del_ganador(busqueda):
    cifras = []
    for k in range(5):  # una cifra por pliegue, la del ganador
        cifras.append(busqueda.cv_results_[f'split{k}_test_score'][busqueda.best_index_])
    return np.array(cifras)

cv_bosque_defecto = cross_val_score(bosque, X_trainval, y_trainval, cv=pliegues, scoring='roc_auc')
comparacion = pd.DataFrame({'logística': cv_logistica,
                            'árbol ajustado': cifras_del_ganador(busqueda_arbol),
                            'bosque por defecto': cv_bosque_defecto,
                            'bosque ajustado': cifras_del_ganador(busqueda_bosque),
                            'boosting ajustado': cifras_del_ganador(busqueda_boosting)})
comparacion.round(3)
"""),
    code("""
tabla_final = pd.DataFrame({'media': comparacion.mean(),
                            'desviación': comparacion.std(ddof=0)})   # ddof=0: la misma desviación de cv_results_
tabla_final['le gana a la logística en'] = comparacion.gt(comparacion['logística'], axis=0).sum()
display(tabla_final.round(3))
display(Markdown(
    f'El bosque ajustado: **{num(tabla_final.loc["bosque ajustado", "media"], 3)}**, y le gana a la '
    f'logística en **{tabla_final.loc["bosque ajustado", "le gana a la logística en"]} de 5** pliegues. '
    f'El bosque por defecto: **{num(tabla_final.loc["bosque por defecto", "media"], 3)}**. El boosting '
    f'ajustado: **{num(tabla_final.loc["boosting ajustado", "media"], 3)}**, en **'
    f'{tabla_final.loc["boosting ajustado", "le gana a la logística en"]} de 5**.'
))
"""),
    md("""
### Paso 4 · Elegir y abrir prueba, una sola vez

La celda trae `elegido = None`: todavía nadie eligió. Escriban `'bosque'`, `'boosting'`, `'arbol'` o
`'logistica'` con la tabla del paso 3 delante, y recién entonces corran las dos celdas que siguen.

<!-- contrato: accion=elegir-modelo -->
"""),
    code("""
elegido = None  # escriban 'bosque', 'boosting', 'arbol' o 'logistica'
"""),
    md("""
Esta es la **única celda que lee prueba**: no elige, informa cuánto rinde el elegido. Los tres
ganadores de las búsquedas ya están entrenados de nuevo con `X_trainval`; la logística se entrena ahora con los
mismos clientes. Prueba se abre **una sola vez**: si ya la abrieron para un modelo, la celda no la abre
para otro, porque cambiar de modelo después de mirarla sería elegir con prueba.

<!-- contrato: accion=abrir-la-prueba -->
"""),
    code("""
nombre_tabla = {'bosque': 'bosque ajustado', 'boosting': 'boosting ajustado', 'arbol': 'árbol ajustado',
                'logistica': 'logística'}
if elegido not in nombre_tabla:
    display(Markdown('Todavía no eligen: escriban `bosque`, `boosting`, `arbol` o `logistica` en la celda anterior.'))
elif len(pruebas_abiertas) > 0 and pruebas_abiertas[0] != elegido:
    display(Markdown(f'Prueba ya se abrió para **{nombre_tabla[pruebas_abiertas[0]]}**: la decisión es esa.'))
else:
    finales = {'bosque': busqueda_bosque.best_estimator_, 'boosting': busqueda_boosting.best_estimator_,
               'arbol': busqueda_arbol.best_estimator_, 'logistica': logistica.fit(X_trainval, y_trainval)}
    auc_prueba = roc_auc_score(y_test, finales[elegido].predict_proba(X_test)[:, 1])  # prueba, acá y solo acá
    if len(pruebas_abiertas) == 0:   # volver a correr con el mismo modelo no cuenta como otra apertura
        pruebas_abiertas.append(elegido)
    display(Markdown(
        f'**{nombre_tabla[elegido].capitalize()}** sobre los {num(len(y_test))} clientes de prueba: ROC-AUC '
        f'**{num(auc_prueba, 3)}**, contra {num(tabla_final.loc[nombre_tabla[elegido], "media"], 3)} de su '
        f'validación cruzada. Prueba también es una sola partición, con su propia suerte: esta cifra se informa, '
        f'pero no se compara contra la validación cruzada de otro modelo.'
    ))
"""),
    md("""
### Paso 5 · Su defensa, en tres líneas

Un modelo es mejor si, con los mismos pliegues, su ventaja supera la desviación. Si gana por poco, decide
lo que se pierde al cambiar: ni el bosque ni el boosting se leen como reglas. Con las cifras de los pasos 3 y 4:

1. Qué modelo eligen:
2. Por cuánto gana o pierde, y en cuántos pliegues:
3. Qué se gana y qué se pierde con él:

# Cierre

## Mejor, y no suerte

Un árbol se lee como reglas, pero cambia con la muestra. El bosque estabiliza lo que predice para
cada cliente; no la cifra de una partición. Para la cifra está la validación cruzada: cinco cifras en
vez de una, y los modelos se comparan con los **mismos** pliegues. `GridSearchCV` elige los
hiperparámetros con esos pliegues, se informan todos los intentos y prueba se abre una sola vez, para el
elegido.

La validación cruzada también cierra una pregunta más vieja: fijar la semilla de la partición vuelve la
cifra reproducible, pero no la libera de la suerte de esa partición. Medir con varias, sí.
"""),
    md(f"""
## Atribución de datos

- **Creador:** Yeh, I. C., & Lien, C. H. (2009)
- **Fuente:** [UCI Machine Learning Repository — *Default of Credit Card Clients*, dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** obra derivada. Se tomaron 6.000 registros y 16 columnas del original de 30.000 ×
  25, con los nombres traducidos al español en `snake_case`, la escolaridad recodificada a etiquetas y
  dos razones derivadas nuevas; [`_prep_riesgo_impago.py`]({PREP_PUBLICO}) documenta cada paso. Las
  columnas de sexo y estado civil se excluyeron deliberadamente.
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


def sin_comprensiones(codigo: list[dict]) -> None:
    """Python básico (docente, 2026-09-09): el material del estudiante no usa comprensiones ni lambda."""
    prohibidos = (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Lambda)
    for celda in codigo:
        for nodo in ast.walk(_arbol("".join(celda["source"]))):
            assert not isinstance(nodo, prohibidos), (
                f"una celda usa {type(nodo).__name__}: {''.join(celda['source'])[:80]!r}")


def contrato(celdas: list[dict]) -> dict:
    """Deriva `metadata.curso_contrato` de la lista FINAL de celdas.

    Se calcula, no se escribe a mano: agregar, quitar o mover una celda lo actualiza solo.
    """
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
    # Prueba se abre UNA vez. Se mide, no se declara: si otra celda la lee, falla acá.
    lectura_prueba = {v: celdas_que_leen(codigo, v) for v in ("X_test", "y_test")}
    assert lectura_prueba == {"X_test": 1, "y_test": 1}, (
        f"prueba la lee más de una celda (o ninguna): {lectura_prueba}")
    return {
        "forma": {
            "markdown": markdown,
            "codigo": len(codigo),
            "llamadas": [llamada(fuente, "train_test_split"), llamada(fuente, "cross_val_score")],
        },
        "acciones": [
            {"marcador": "fijar-base", "asignacion": "datos"},
            {"marcador": "declarar-positiva", "asignacion": "y"},
            {"marcador": "partir", "codigo": "train_test_split"},
            {"marcador": "arbol-dos", "codigo": "DecisionTreeClassifier"},
            {"marcador": "arbol-libre", "asignacion": "arbol_libre"},
            {"marcador": "grupo-referencia", "asignacion": "X_ref"},
            {"marcador": "veinte-particiones", "codigo": "RandomForestClassifier"},
            {"marcador": "validar-cruzado", "codigo": "cross_val_score"},
            {"marcador": "curva-u", "codigo": "validation_curve"},
            {"marcador": "buscar-arbol", "codigo": "GridSearchCV"},
            {"marcador": "contar-ajustes", "asignacion": "ajustes_bosque"},
            {"marcador": "buscar-bosque", "asignacion": "busqueda_bosque"},
            {"marcador": "buscar-boosting", "codigo": "HistGradientBoostingClassifier"},
            {"marcador": "comparar", "asignacion": "comparacion"},
            {"marcador": "elegir-modelo", "asignacion": "elegido"},
            {"marcador": "abrir-la-prueba", "asignacion": "auc_prueba"},
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
