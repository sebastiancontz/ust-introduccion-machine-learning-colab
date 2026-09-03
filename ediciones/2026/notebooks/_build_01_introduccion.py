#!/usr/bin/env python3
"""Construye notebooks/01-introduccion.ipynb.

POR QUÉ EXISTE ESTE GENERADOR, Y POR QUÉ LLEGÓ TARDE: el notebook de la clase 1 se armó con un
script que quedó en un scratchpad temporal, así que durante meses fue el único artefacto sin fuente
versionada. Cada corrección había que aplicarla al `.ipynb` a mano, y una vez ya se revirtieron dos
en silencio. Este archivo se reconstruyó desde el `.ipynb` publicado el 2026-08-11, celda por celda,
para cerrar esa deuda: de acá en adelante el `.ipynb` NO se edita a mano.

QUÉ CAMBIÓ AL RECONSTRUIRLO, y nada más que eso:
  · Se quitó el párrafo que explicaba el respaldo Colab/local del dataset. Es fontanería del
    notebook: el código de abajo se lee solo y al curso no le aporta nada. Salió también de las
    clases 2 y 3, y la plantilla del template ya lo prohíbe.
  · `figura_base64` exige `alt`, y el logo de la cabecera lo lleva. Sin él la figura no existe para
    quien usa lector de pantalla y `nbconvert` lo reclama al exportar. Es el mismo cambio que ya
    tenía la clase 3.
  · El diagrama se embebe desde `assets/ilustraciones/taxonomia-aprendizaje.svg`, que es su fuente.
    El .ipynb traía una copia base64 anterior al `<style>` de Noto Sans que el SVG ganó después.

FASE 5E (2026-08-11, primera revisión de consistencia derivada que pasa esta clase):
  · El párrafo que lee el panel de meses de mora afirmaba cinco magnitudes —15 %, 33 %, 66 % y
    grupos de 5, 2 y 2— que ninguna celda calculaba: la única salida vecina era un gráfico de
    CONTEOS. Ahora las imprime una celda y el texto cita los valores exactos que muestra. De paso
    se cumple la promesa «hay que mirar la proporción dentro de cada grupo», que el notebook hacía
    y no cumplía: el ejercicio final agrupa por nivel de educación, no por meses de mora.
  · Esa misma celda introduce `groupby` y `normalize=True` con su explicación en Markdown ANTES de
    que el ejercicio final se los pida. Antes el esqueleto del ejercicio los usaba sin haberlos
    nombrado, y el chequeo de APIs no lo veía porque vive dentro de una celda de Markdown.
  · Los tipos se imprimen con `type(...).__name__`. La ruta completa de la clase cambió entre
    pandas 2 y 3, justo en la celda cuyo objetivo es leer el tipo, y Colab no siempre trae la misma
    versión que el entorno local. Por lo mismo se declara en Markdown que el texto aparece como
    `object` o como `str` según la versión.
  · El ejercicio final pide también los conteos por nivel, porque uno de los niveles tiene once
    clientes y su porcentaje es exactamente la trampa que la sección anterior acaba de enseñar.

CIFRAS: las que el texto afirma están verificadas contra `datasets/morosidad_cartera.csv`. Si el
dataset se regenera hay que volver a verificarlas, porque acá se afirman en el texto.

Uso:  python ediciones/2026/notebooks/_build_01_introduccion.py
      (después: ejecutar el notebook para dejarlo con salidas, ver el README del repo)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "01-introduccion.ipynb"
SVG = REPO_ROOT / "assets" / "ilustraciones" / "taxonomia-aprendizaje.svg"


def figura_base64(path: Path, alt: str, width: int = 900) -> str:
    """Diagrama embebido como data URI: el notebook queda autocontenido y no depende de Pages.

    `alt` es obligatorio: sin él la figura no existe para quien usa lector de pantalla, y
    `nbconvert` lo reclama al exportar.
    """
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<p align="center"><img src="data:image/svg+xml;base64,{b64}" '
            f'width="{width}" alt="{alt}"></p>')


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def con_ids(celdas: list[dict]) -> list[dict]:
    """nbformat 4.5+ exige un id por celda. Deterministas: regenerar no produce ruido en el diff."""
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
<p>Semana 01: ¿Qué es Machine Learning?</p>
</div>
</div>
"""),
    md("""
# 01 · Introducción al Machine Learning

**Qué van a hacer acá:** primero terminar el taller de clasificación de casos, y después recorrer
las librerías del curso sobre un caso real de morosidad, viendo qué aporta cada una.

Este notebook tiene tres partes:

1. **El taller** — tres situaciones de gestión para encuadrar. No necesita que el código corra.
2. **El recorrido** — cargar el caso y pasar por pandas, NumPy, Matplotlib, Seaborn y scikit-learn.
3. **El cierre** — el mismo caso, encuadrado de tres maneras distintas.

Si es la primera vez que usan Colab: para ejecutar una celda, hagan clic en ella y presionen
**Shift + Enter**. Las celdas se ejecutan **en orden**; si algo falla, casi siempre es porque se
saltaron una.
"""),
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas, así que la celda siguiente termina en segundos. Está
igual porque el notebook también tiene que correr fuera de Colab, y porque dejar escrito de qué
depende un análisis es parte de que sea reproducible.
"""),
    code("""
%%capture
!pip install -q pandas numpy matplotlib seaborn scikit-learn
"""),
    md("""
## Cargar datos

Dos métodos aparecen acá por primera vez: `read_csv`, que lee un archivo separado por comas y devuelve una tabla, y `head`, que muestra sus primeras filas para confirmar de un vistazo que llegó lo que se esperaba.
"""),
    code("""
import os
import pandas as pd

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

df = pd.read_csv(BASE + 'morosidad_cartera.csv')
print('Filas y columnas:', df.shape)
df.head()
"""),
    md("""
Son 600 clientes de crédito de consumo. Cada fila es un cliente, y la última columna dice si
**incumplió el pago del mes siguiente**. El diccionario de datos completo está en
[`datasets/README.md`](https://github.com/sebastiancontz/ust-introduccion-machine-learning-colab/blob/main/ediciones/2026/datasets/README.md).

> **Origen de los datos.** Adaptados de *Default of Credit Card Clients* del
> [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
> de Yeh, I. C. y Lien, C. H. (2009), bajo licencia
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Versión modificada: submuestra de
> 600 filas y 8 columnas, traducidas y recodificadas.
"""),
    md("""
---

# Parte 1 · El taller

En clase encuadramos tres casos juntos. Acá van los tres que quedan. Para cada uno, las mismas
cuatro preguntas **en orden**:

1. ¿Se puede abordar **aprendiendo de datos históricos**? Si no, ¿por qué no?
2. Si sí: ¿los datos traen la **respuesta conocida**? Con etiqueta es **supervisado**; sin etiqueta,
   **no supervisado**.
3. Si es supervisado: ¿la etiqueta es una cantidad o una categoría? **Regresión** o **clasificación**.
4. ¿Cuál sería exactamente esa etiqueta, y estaría **disponible al momento de decidir**?

Ojo con la primera: si la respuesta es *no*, el ejercicio **termina ahí**. No fuercen las otras tres.

Lo que importa es la justificación, no la etiqueta que le pongan.
"""),
    md("""
### Caso A

> *"Queremos anticipar cuáles de las solicitudes de crédito que aprobamos van a caer en mora en los
> próximos tres meses, para priorizar a quién le hacemos seguimiento telefónico."*

**Su respuesta** (doble clic para editar esta celda):

1.
2.
3.
4.
"""),
    md("""
### Caso B

> *"Necesitamos calcular el recargo por mora que le corresponde a cada factura vencida, según lo que
> establece el contrato."*

**Su respuesta:**

1.
2.
3.
4.
"""),
    md("""
### Caso C

> *"Queremos estimar cuánto va a facturar cada cliente el mes que viene, para dimensionar el equipo
> de cobranza."*

**Su respuesta:**

1.
2.
3.
4.
"""),
    md("""
> **Pista para el caso C:** presten atención a la pregunta 4. Es la que más cuesta y la que más
> importa.
"""),
    md("""
---

# Parte 2 · El recorrido del ecosistema

Cinco librerías, un caso. La idea no es aprender a usarlas hoy —eso es el resto del semestre— sino
ver **qué aporta cada una**.

Esta parte va en tres tiempos: primero **miramos** celdas ya resueltas, después las **revisamos
juntos** prediciendo qué va a salir antes de ejecutar, y al final hay **una para ustedes**.

<!-- Liberación gradual (interno, NO decirlo en pantalla): "Cargar y mirar" es el yo lo hago;
     "Revisémoslo juntos" es el lo hacemos, con predicción de salida y una modificación guiada;
     "Ahora ustedes" es el lo haces, con celda en blanco. -->
"""),
    md("""
## Cargar y mirar

pandas es la mesa de trabajo: carga la tabla y permite inspeccionarla. Tres preguntas que uno le
hace a cualquier dataset nuevo, antes de cualquier otra cosa. Estas celdas ya están resueltas:
síganlas mientras las recorremos.

Y tres métodos más, en las celdas que siguen: `info` resume la tabla —cuántas filas tiene, qué tipo
tiene cada columna y cuántos valores no vacíos trae—, `describe` calcula de un golpe los estadísticos
básicos de cada columna numérica, y `value_counts` cuenta cuántas veces aparece cada valor de una
columna.

Según la versión de pandas, las columnas de texto aparecen como `object` o como `str`. Significan lo
mismo: **no es un número**.
"""),
    code("""
# ¿Qué tipo de dato tiene cada columna? ¿Falta algo?
df.info()
"""),
    code("""
# ¿Cómo se distribuyen las columnas numéricas?
df.describe()
"""),
    code("""
# ¿Cuántos clientes incumplieron?
df['incumplio_pago'].value_counts()
"""),
    md("""
**Miren la tabla de `describe()` con atención antes de seguir.**

El mínimo de `monto_facturado_mes` es **negativo**. ¿Un monto facturado negativo?

No es un error de carga: significa que el cliente tenía **saldo a favor**. Pero el punto es otro, y
es el punto de esta celda: *alguien tiene que darse cuenta*. Un modelo no se va a quejar de un valor
raro; lo va a usar igual. Mirar los datos antes de modelar es trabajo, no trámite — y es la clase 3
completa.
"""),
    md("""
---

## Revisémoslo juntos

De acá en adelante, **antes de ejecutar cada celda, respondan la pregunta**. Después comparan con lo
que salió. Equivocarse acá no cuesta nada y es la forma más rápida de que algo quede.

### NumPy · calcular sobre todo a la vez

En las slides dijimos que pandas **se apoya en NumPy** por debajo. No hace falta creerlo: se puede
verificar.

La brecha entre lo facturado y lo pagado es una resta, y se aplica a las 600 filas sin ningún bucle.

> **Antes de ejecutar:** al restar dos columnas de un DataFrame, ¿qué tipo de objeto creen que
> devuelve pandas? ¿Y de qué tipo son los valores que hay dentro?
"""),
    code("""
import numpy as np

brecha = df['monto_facturado_mes'] - df['monto_pagado_mes']

# __name__: el nombre pelado de la clase. La ruta completa cambia entre versiones de pandas
print('Tipo del objeto que devolvió pandas:', type(brecha).__name__)
print('Y por debajo, sus valores son:', type(brecha.values).__name__)
print()
# el promedio, calculado por NumPy sobre el arreglo que hay debajo de la Series
print('Brecha promedio:', round(float(np.mean(brecha.values)), 1))
print('Clientes que pagaron más de lo facturado:', int(np.sum(brecha.values < 0)))
"""),
    md("""
Ahí está: pandas devolvió una `Series`, pero sus valores son un `ndarray` de **NumPy**. Esa es la
capa de abajo, haciendo el cálculo.

Eso es lo que hace que operar 600 filas —o 600 mil— sea instantáneo en vez de un bucle que hay que
esperar.
"""),
    md("""
### Matplotlib y Seaborn · mirar la forma

Un número resume; un gráfico muestra la forma. La pregunta acá es concreta: **¿se ven distintos los
clientes que incumplieron de los que no?** Si no se vieran distintos en nada, no habría patrón que
aprender.

> **Antes de ejecutar:** ¿en cuál de las dos variables esperan ver más diferencia entre quienes
> incumplieron y quienes no: en el cupo de crédito o en los meses de mora? ¿Por qué?
"""),
    code("""
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# el color se define UNA vez, con la paleta tab10 de Tableau: los gráficos de más abajo
# la heredan sin repetir códigos de color en cada llamada
sns.set_theme(style='whitegrid', palette='tab10')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# order y hue_order fijos en AMBOS paneles: sin esto cada gráfico asigna los
# colores según el orden de aparición del dato y el mismo valor sale de distinto
# color en cada panel, que en una figura donde el color codifica la clase se lee mal
ORDEN = ['no', 'si']

sns.boxplot(data=df, x='incumplio_pago', y='limite_credito', ax=axes[0],
            order=ORDEN, hue='incumplio_pago', hue_order=ORDEN, legend=False)
axes[0].set_title('Cupo de crédito')
axes[0].set_xlabel('¿Incumplió?')
axes[0].set_ylabel('Límite de crédito')

sns.countplot(data=df, x='meses_mora', ax=axes[1], hue='incumplio_pago', hue_order=ORDEN)
axes[1].set_title('Meses de mora')
axes[1].set_xlabel('Meses de atraso')
axes[1].set_ylabel('Clientes')
axes[1].legend(title='¿Incumplió?')

plt.tight_layout()
plt.show()
"""),
    md("""
A la izquierda, quienes incumplieron tienden a tener un **cupo más bajo**. Se superponen bastante,
así que el cupo por sí solo no alcanza para decidir.

A la derecha, cuidado con cómo se lee: el panel muestra **conteos**, no proporciones. Las barras
altas son las de los grupos numerosos, no las de mayor riesgo. Para comparar riesgo hay que mirar la
proporción **dentro de cada grupo**, y eso el gráfico no lo dice. Calculémosla.

Dos piezas nuevas: `groupby` parte la tabla en grupos según los valores de una columna y repite la
misma cuenta en cada grupo por separado, y el argumento `normalize=True` hace que `value_counts`
devuelva proporciones en vez de conteos.
"""),
    code("""
riesgo = df.groupby('meses_mora')['incumplio_pago'].value_counts(normalize=True)

print('Clientes en cada grupo:')
print(df['meses_mora'].value_counts().sort_index())
print()
print('Porcentaje de cada grupo que incumplió:')
print((riesgo * 100).round(1))
"""),
    md("""
Ahora sí se puede comparar. **Hasta dos meses de atraso el riesgo sube fuerte:** 15,2 %, 32,9 % y
66,0 %. De tres en adelante los grupos tienen 5, 2 y 2 clientes, y con esos números no se lee nada
— de hecho con `meses_mora = 4` ninguno incumplió y con 3 y 5 incumplieron todos. Un grupo chico
produce porcentajes extremos que no significan nada, y por eso el conteo va siempre al lado de la
proporción.

**Ninguna de las dos cosas es una causa.** Que el cupo sea bajo no *causa* el incumplimiento: es
probable que ambos respondan a algo anterior. Es exactamente la distinción de la segunda mitad de la
clase, ahora sobre datos.

Seaborn hizo estos gráficos en pocas líneas, y por debajo usó Matplotlib. Otra vez la idea de capas.

**Prueben esto:** en la celda del gráfico, cambien `limite_credito` por `monto_pagado_mes` y vuelvan
a ejecutar. ¿Se separan más o menos los dos grupos que con el cupo?
"""),
    md("""
### scikit-learn · el piso contra el que comparar

Todavía no vamos a entrenar un modelo de verdad: eso arranca en la unidad 2. Pero scikit-learn tiene
algo que aportar **hoy**, y es lo primero que hay que hacer en cualquier proyecto: establecer **el
piso**.

La estrategia más tonta posible: predecir siempre lo mismo, que nadie incumple.

> **Antes de ejecutar, arriesguen un número:** ¿qué porcentaje de aciertos creen que va a tener ese
> "modelo" que no mira ninguna variable? ¿Y a cuántos morosos creen que va a detectar?
"""),
    code("""
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

X = df[['edad', 'limite_credito', 'meses_mora', 'monto_facturado_mes']]
y = df['incumplio_pago']

piso = DummyClassifier(strategy='most_frequent')
piso.fit(X, y)
prediccion = piso.predict(X)

print('Exactitud del piso:', round(accuracy_score(y, prediccion), 3))
print('Morosos que detectó:', int((prediccion == 'si').sum()), 'de', int((y == 'si').sum()))
"""),
    md("""
Ahí está la trampa del chequeo de la clase, ahora sobre nuestros datos.

Este "modelo" **no mira ninguna variable**: contesta lo mismo siempre. Y aun así acierta más de tres
de cada cuatro veces, simplemente porque la mayoría de los clientes paga. Detecta **cero** morosos,
que es justo lo único que nos importaba.

Dos cosas que se llevan de acá:

- **Una exactitud alta, sola, no dice nada.** Hay que preguntar contra qué se compara.
- **Cualquier modelo que construyamos tiene que ganarle a esto.** Si no le gana, no sirve, por más
  sofisticado que sea.

Y una advertencia honesta: esto está **incompleto a propósito**. Lo evaluamos con los mismos datos
con que lo ajustamos, y eso es exactamente lo que no hay que hacer. Falta separar una parte de los
datos para evaluar, y eso es la clase 6. Hoy quedémonos con el piso.
"""),
    md("""
---

## Ahora ustedes

Una sola tarea, con lo que acabamos de ver.

**Pregunta:** ¿la proporción de incumplimiento es igual en todos los niveles de educación?

Les dejo el esqueleto. La pieza que falta es la columna por la que hay que agrupar:

```python
df.groupby('___')['incumplio_pago'].value_counts(normalize=True)
```

Escriban su versión en la celda de abajo. Si sale un error, léanlo: casi siempre dice exactamente
qué nombre no encontró. Y miren también **cuántos clientes** tiene cada nivel, como recién: la
proporción sola no basta.

> **Y la pregunta que importa más que el código:** si encuentran una diferencia entre niveles,
> ¿alcanza para decir que el nivel educativo **causa** el incumplimiento?
"""),
    code("""
# escriban acá su respuesta
"""),
    md("""
Y antes de la respuesta, un aviso sobre lo que van a ver: uno de los niveles es un grupo muy chico.
Su porcentaje va a llamar la atención y no quiere decir nada, por lo mismo que acabamos de ver con
los meses de mora.

Sobre la última pregunta: no, no alcanza. Encontrar que dos cosas van juntas en los datos no dice
cuál mueve a cuál, ni si hay algo detrás moviendo a las dos. Es el punto central de la clase, ahora
sobre una tabla que ustedes mismos calcularon.
"""),
    md("""
---

# Parte 3 · El mismo caso, tres encuadres

Trabajamos un solo archivo. Según qué columna elijamos como respuesta, es un problema distinto:

| Si la etiqueta es… | El problema es… | La pregunta de gestión |
|:--|:--|:--|
| `incumplio_pago` | clasificación | ¿a quién le hago seguimiento? |
| `monto_facturado_mes` | regresión | ¿cuánto se va a facturar? |
| ninguna | agrupamiento | ¿qué perfiles de pago existen en la cartera? |

**Quien define la etiqueta define el problema.** Eso es la clase 2.

""" + figura_base64(
        SVG, width=900,
        alt="Árbol de tipos de aprendizaje: machine learning se abre en aprendizaje supervisado, "
            "cuyos datos traen la respuesta, y no supervisado, cuyos datos no la traen; el "
            "supervisado se divide en clasificación y regresión, y el no supervisado en "
            "agrupamiento; cada hoja muestra una nube de puntos que ilustra la forma de sus datos. "
            "El aprendizaje por refuerzo aparece marcado como fuera del alcance del curso")),
    md("""
## Cierre

- **Qué hicimos:** encuadramos casos de gestión, y recorrimos las cinco librerías sobre un caso real
  viendo qué aporta cada una.
- **Qué decisión permite tomar:** con una estimación de riesgo por cliente se puede **ordenar** la
  cartera y gastar un presupuesto acotado de cobranza en los casos más probables.
- **Qué límites hay que comunicar:** el modelo no dice **por qué** un cliente incumple, y no dice
  qué pasaría si intervenimos. Y una exactitud alta puede no significar nada.

### Para la próxima

En la clase 2 vamos a definir bien la etiqueta: qué cuenta como incumplimiento, en qué plazo, y qué
información estaba realmente disponible en el momento de decidir.
"""),
    md("""
## Atribución de datos

- **Creador:** Yeh, I. C. y Lien, C. H. (2009)
- **Fuente:** [UCI Machine Learning Repository — *Default of Credit Card Clients*, dataset 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** submuestra de 600 filas y 8 de las 25 columnas, renombradas al español y con
  algunas categorías recodificadas. El detalle está en el `README.md` de `datasets/`.
"""),
]


def main() -> None:
    if not SVG.exists():
        raise SystemExit(f"falta el diagrama {SVG}")
    nb = {
        "cells": con_ids(CELDAS),
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_md = sum(1 for c in CELDAS if c["cell_type"] == "markdown")
    print(f"{SALIDA.relative_to(REPO_ROOT)}: {len(CELDAS)} celdas "
          f"({n_md} markdown / {len(CELDAS) - n_md} código)")


if __name__ == "__main__":
    main()
