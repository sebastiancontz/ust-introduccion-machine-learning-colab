#!/usr/bin/env python3
"""Construye notebooks/repaso-solemne-01.ipynb: las dos actividades del repaso de la Unidad 1.

POR QUÉ ES UNA PLANTILLA Y NO LLEVA SALIDAS: sus celdas de código traen espacios `?????` que el
estudiante completa, así que no se pueden ejecutar tal como están —fallarían con un error de
sintaxis o de nombre—. Por eso el notebook declara `metadata.curso_plantilla: true`, que es el
único caso en que el repo acepta un notebook sin salidas guardadas. No lleva `curso_contrato`:
los contratos verifican notebooks de clase ejecutados, y acá no hay salida que verificar.

POR QUÉ EL ARCHIVO NO ES EL DE LA EVALUACIÓN: la parte práctica de la Solemne 1 trabaja sobre otro
dataset. Si el repaso usara el mismo, practicar se convertiría en memorizar sus cifras. Lo que se
repite acá es la FORMA de las preguntas —pasos de código con espacios marcados y pasos de
interpretación escrita—, no los datos.

Y ESO NO SE DICE EN EL MATERIAL VISIBLE (docente, 2026-09-10): la forma del instrumento es sorpresa.
El notebook y la guía describen cada actividad por lo que hay que hacer en ella, nunca por su
parecido con la evaluación. Esta nota es interna; no la promuevas a una celda.

CIFRAS: ninguna magnitud de la solución se escribe en este archivo. Las respuestas comentadas, con
sus cifras medidas, viven en `guias_estudio/repaso-solemne-01.md`, que es la fuente única de la
página y del PDF del repaso. Acá solo van los enunciados y los andamiajes.

COMPATIBILIDAD: toda lista de valores únicos se imprime con `tolist()`. Sin él, la envoltura con
que NumPy muestra un escalar viaja a la salida —`[np.int64(0)]` en vez de `[0]`— y cambia entre
versiones, así que el estudiante ve algo distinto en Colab que en la sala. Detectado ejecutando el
notebook resuelto, no leyéndolo.

APIs: todo lo que usan las dos actividades es de la Unidad 1. `named_transformers_` y
`get_feature_names_out` quedaron deliberadamente afuera aunque resolverían el paso 2 de la
actividad 2 en una línea: llegan con la clase 6, que ya es Unidad 2, y el ancho de cada ruta se
puede razonar con lo que la unidad sí instaló.

Uso:  python ediciones/2026/notebooks/_build_repaso_solemne_01.py
"""
from __future__ import annotations

import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "repaso-solemne-01.ipynb"

LOGO = ("https://sebastiancontz.github.io/ust-introduccion-machine-learning"
        "/assets/logo-ust.svg")
GUIA = ("https://sebastiancontz.github.io/ust-introduccion-machine-learning"
        "/ediciones/2026/guias/repaso-solemne-01.html")
DATASET = ("https://raw.githubusercontent.com/sebastiancontz/"
           "ust-introduccion-machine-learning-colab/main/ediciones/2026/"
           "datasets/absentismo_laboral.csv")


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def respuesta(paso: int) -> dict:
    return md(f"""
### Respuesta escrita — paso {paso}

_Escriban su respuesta acá._
""")


CELDAS = [
    md(f"""
<div style="display:flex; align-items:center; gap:18px; text-align:left">
<img src="{LOGO}" width="100" alt="Logo de la Universidad Santo Tomás">
<div>
<p>Ingeniería en Información y Control de Gestión</p>
<p>Facultad de Economía y Negocios</p>
<p>Introducción a Machine Learning</p>
<p>Repaso de la Unidad 1: Fundamentos de Machine Learning</p>
</div>
</div>
"""),

    md(f"""
# Repaso · Actividades de la Unidad 1

Dos actividades para practicar con datos los conceptos de la Unidad 1. Cada paso se responde
escribiendo o completando los espacios marcados `?????`.

Las dos trabajan sobre `absentismo_laboral.csv`, que **no** es el archivo que verán en la
evaluación. Es a propósito: lo que hay que llevarse es el procedimiento, no las cifras de un
archivo.

**Cómo trabajar.** Hagan una copia de este notebook en su propia unidad. Resuelvan primero sin
consultar. Los **criterios de éxito** y las **respuestas comentadas** de las dos actividades están
en la [guía del repaso]({GUIA}), plegadas: ábranlas después de intentarlo.

Cada decisión va con su justificación escrita. Una columna bien descartada sin el criterio que la
descarta no muestra el razonamiento que la unidad pide.
"""),

    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas, así que la celda siguiente termina en segundos.
Está igual porque el notebook también tiene que correr fuera de Colab.
"""),

    code("""
%%capture
!pip install -q pandas numpy scikit-learn
"""),

    # ── Actividad 1 ───────────────────────────────────────────────────────────────────────────
    md("""
---

## Actividad 1 · Formular y fijar el piso

**Nivel básico. Cuatro pasos.**

Una empresa de mensajería tiene un problema de cobertura de turnos: cuando alguien falta, el
despacho se reorganiza a última hora y ese día se atrasan las entregas. La jefatura de operaciones
quiere anticipar **cuántas horas de ausencia** va a tener cada episodio para reorganizar el turno
antes de que el día empiece, en vez de a mitad de mañana. Les piden formular el problema antes de
que nadie entrene un modelo.

El archivo `absentismo_laboral.csv` tiene registros de ausencias del personal, con las variables del
empleado y del período de cada ausencia. La celda siguiente lo carga en la variable `datos`.
"""),

    md("""
**Preparación.** Ejecuten esta celda antes de responder: carga el archivo y muestra su estructura.
"""),

    code(f"""
import numpy as np
import pandas as pd

RUTA = (
    "{DATASET}"
)
datos = pd.read_csv(RUTA)
datos.info()
"""),

    md("""
### Paso 1 · La ficha de formulación

Respondan las tres preguntas, una por línea. Cada una se sostiene con lo que el archivo muestra, no
con lo que su nombre sugiere.

1. ¿Qué representa un registro de este archivo? Sosténganlo con dos conteos.
2. ¿Cuál es la variable objetivo y en qué unidad está medida?
3. ¿En qué momento tiene que estar lista la predicción para que la jefatura alcance a reorganizar
   el turno?
"""),

    respuesta(1),

    md("""
### Paso 2 · El piso

El baseline es la referencia contra la que se compara todo modelo. Acá se usa la regla más simple:
predecir siempre el mismo valor para todos los episodios.

1. Completen la celda. Usen la **media** como valor constante, para que todo el curso reporte la
   misma referencia.
2. Reporten el error del baseline como el **promedio del error absoluto, en horas por episodio**.
3. Expliquen en una o dos frases qué le dice esa cifra a quien tiene que reorganizar el turno, y
   qué **no** le dice sobre qué episodio cubrir primero.
"""),

    code("""
# complete los espacios marcados abajo
objetivo = datos["?????"]

# el baseline predice siempre el mismo valor para todos los episodios
piso = objetivo.?????()

error_promedio = np.mean(np.abs(objetivo - piso))
print(f"baseline: {piso:.2f} horas · error promedio: {error_promedio:.2f} horas")
"""),

    respuesta(2),

    md("""
### Paso 3 · Un registro que no es lo que parece

La columna `falta_disciplinaria` marca un subconjunto de registros. Antes de seguir usando el piso
del paso 2, hay que saber qué son esos registros.

1. Completen la celda: cuenten cuántos son y miren con qué horas y con qué motivo aparecen.
2. Completen la segunda parte, que recalcula el mismo piso dejando fuera esos registros, y reporten
   las dos cifras nuevas.
3. Expliquen qué son esos registros, si pertenecen o no a la unidad de observación que declararon
   en el paso 1, y **qué cambió realmente** entre las dos cifras del piso. Cuidado con la lectura
   fácil: la pregunta no es si el error bajó.
"""),

    code("""
# complete los espacios marcados abajo

# (a) que son los registros marcados
disciplinarios = datos["falta_disciplinaria"].eq("si")
print(f"registros marcados: {disciplinarios.sum()}")
print("horas:", sorted(datos.loc[disciplinarios, "?????"].unique().tolist()))
print("motivos:", datos.loc[disciplinarios, "motivo_ausencia"].unique().tolist())

# (b) el mismo piso, ahora sin ellos
episodios = datos.loc[~disciplinarios]
piso_episodios = episodios["horas_ausencia"].?????()
error_episodios = np.mean(np.abs(episodios["horas_ausencia"] - piso_episodios))
print(f"{len(episodios)} registros · baseline: {piso_episodios:.2f} horas "
      f"· error promedio: {error_episodios:.2f} horas")
"""),

    respuesta(3),

    md("""
### Paso 4 · Dos columnas que quedan fuera, por dos motivos distintos

La jefatura propone dos columnas para el modelo:

1. `motivo_ausencia`, que registra por qué faltó la persona.
2. `indice_masa_corporal`, que el sistema de salud ocupacional ya tiene registrado para cada
   empleado desde antes.

Para cada una, decidan si entra a la ficha de formulación y **nombren el criterio** que la resuelve.
Los dos criterios **no** son el mismo. Digan además qué otras columnas del archivo caen bajo el
mismo criterio que la segunda.
"""),

    respuesta(4),

    # ── Actividad 2 ───────────────────────────────────────────────────────────────────────────
    md("""
---

## Actividad 2 · Cada columna entra representada, o no entra

**Nivel avanzado. Cuatro pasos.**

Una empresa de mensajería quiere anticipar cuántas horas de ausencia va a tener cada episodio, para
reorganizar el turno antes de que el día empiece. La revisión de calidad del archivo ya se hizo. Lo
que falta es la decisión que les encargan a ustedes: **cómo entra cada columna al preprocesador**, y
qué pasa con las que no entran.

Acá **no se entrena ningún modelo y no se reporta ninguna cifra de error**: lo que se practica es la
decisión de representación y cómo se sostiene con el archivo a la vista.

La celda de preparación deja los datos en `datos`, declara el objetivo en `OBJETIVO` y entrega el
diccionario `ORDEN_EDUCACION`, que es del negocio y no se descubre en los datos.
"""),

    md("""
**Preparación.** Ejecuten esta celda antes de responder.
"""),

    code(f"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA = (
    "{DATASET}"
)
datos = pd.read_csv(RUTA)

# el orden de nivel_educacion viene dado: es del negocio, no se descubre en los datos
ORDEN_EDUCACION = {{
    "media": 0,
    "universitaria": 1,
    "posgrado": 2,
    "magister o doctorado": 3,
}}

# la variable objetivo, que el enunciado ya declara
OBJETIVO = "horas_ausencia"

print(f"{{len(datos)}} registros y {{datos.shape[1]}} columnas")
"""),

    md("""
### Paso 1 · Repartir las columnas en las tres rutas

El preprocesador ya viene armado con sus tres rutas: una codifica, una escala y una deja pasar la
columna ya numerada. Lo que falta es la decisión de representación: qué columna recibe qué
tratamiento. Ninguna de las tres listas está completa.

1. Completen `NOMINALES` con las **dos** categóricas sin orden que describen **cuándo** ocurrió la
   ausencia.
2. Completen `ORDINAL` con la única columna de texto cuyo orden es **real**, que es la que
   `ORDEN_EDUCACION` sabe numerar.
3. Completen `NUMERICAS` con **cuatro** columnas numéricas del empleado: su edad, su antigüedad,
   cuántos hijos tiene y a qué distancia vive del trabajo.
4. Ejecuten la celda y reporten las **dos** cifras que imprime: cuántas columnas entran al
   preprocesador y cuántas salen. Digan además cuántas columnas aporta cada ruta a la salida, y por
   qué solo una de las tres ensancha la matriz.

El preprocesador recibe **todas** las predictoras candidatas: la celda le entrega el archivo sin la
variable objetivo y sin la versión de texto de la ordinal, que ya quedó numerada. La ruta `"ord"`
usa `"passthrough"` porque esa columna entra sin otra transformación.
"""),

    code("""
# complete los espacios marcados abajo
NOMINALES = ["?????", "?????"]
ORDINAL = "?????"
NUMERICAS = ["?????", "?????", "?????", "?????"]

datos["educacion_num"] = datos[ORDINAL].map(ORDEN_EDUCACION)

preprocesador = ColumnTransformer([
    ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINALES),
    ("num", StandardScaler(), NUMERICAS),
    ("ord", "passthrough", ["educacion_num"]),
])

# el preprocesador recibe todas las predictoras candidatas, no solo las nombradas arriba
X = datos.drop(columns=[OBJETIVO, ORDINAL])
matriz = preprocesador.fit_transform(X)

print(f"entran {X.shape[1]} columnas y salen {matriz.shape[1]}")
for columna in NOMINALES:
    print(f"  {columna}: {datos[columna].nunique()} categorías")
"""),

    respuesta(1),

    md("""
### Paso 2 · Lo que el preprocesador descartó sin aviso

Entraron más columnas de las que salieron. El valor por defecto de `remainder` es `"drop"`: toda
columna que llega al preprocesador y no aparece en ninguna de las rutas **se descarta, sin aviso y
sin error**.

1. Completen la celda con la columna nombrada que falta en la lista, y reporten cuántas columnas se
   descartaron.
2. Comprueben que el desglose por ruta que imprime la celda coincide con la cifra de salida del
   paso 1.
3. Separen la lista de descartadas en **dos grupos, con al menos tres columnas en cada uno**: las
   que corresponde dejar fuera, y las que se **perdieron por omisión** aunque estaban disponibles
   en el momento de decidir y eran utilizables.
4. Escriban el motivo de cada columna del primer grupo. Los motivos **no son todos el mismo**: hay
   un identificador, hay una columna temporal, hay una que se conoce después del hecho, hay una que
   vale exactamente cuando el objetivo vale cero, hay variables que no son legítimas para fundar
   una decisión laboral y hay cifras que no describen al empleado. Si alguno de esos motivos les
   parece discutible, díganlo y sostengan de qué depende.
"""),

    code("""
# complete el espacio marcado abajo
nombradas = NOMINALES + NUMERICAS + ["?????"]

sin_nombrar = [columna for columna in X.columns if columna not in nombradas]

print(f"{len(sin_nombrar)} columnas entraron y el preprocesador las descartó sin aviso:")
for columna in sin_nombrar:
    print(" ", columna)

# cuantas columnas aporta cada ruta a la salida
del_one_hot = sum(datos[columna].nunique() for columna in NOMINALES)
print(f"\\none-hot: {del_one_hot} · escaladas: {len(NUMERICAS)} · ordinal: 1 "
      f"· total: {del_one_hot + len(NUMERICAS) + 1}")
"""),

    respuesta(2),

    md("""
### Paso 3 · Una columna numérica que en realidad es categórica

`mes` es el mes calendario de la ausencia. Está disponible desde antes de que el mes empiece, así
que es una predictora legítima, y ninguna de las tres rutas del paso 1 la nombró.

1. Numerar una categórica afirma **dos** cosas distintas sobre ella. Digan cuáles son las dos, y
   cuál de las dos se puede verificar con este archivo y cuál no.
2. Ejecuten la celda: hay un dato de `mes` que decide el tratamiento y que no se ve en el
   diccionario de datos. Digan qué es y qué hay que hacer con él.
3. Decidan un tratamiento para `mes` —entra como numérica tal como viene, o entra con one-hot— y
   sostengan la decisión.
4. Digan con cuántas columnas queda la salida del paso 1 bajo cada uno de los dos tratamientos.
"""),

    code("""
print("valores distintos de mes:", sorted(datos["mes"].unique().tolist()))
print("registros por valor:")
print(datos["mes"].value_counts().sort_index().to_string())
"""),

    respuesta(3),

    md("""
### Paso 4 · Alta cardinalidad, identificadores y un motivo que no es ninguno de los dos

`motivo_ausencia` e `id_empleado` quedaron fuera del preprocesador, y no por el mismo motivo.
Ejecuten la celda y respondan.

1. Digan con cuántas columnas quedaría la salida del paso 1 si `motivo_ausencia` entrara con
   one-hot, y por qué ese ancho es un problema para un modelo que estima un coeficiente por
   columna. Miren también con cuántos registros aparece cada categoría.
2. Nombren la salida que la unidad propone para aliviar la alta cardinalidad, y digan qué se pierde
   al aplicarla.
3. Digan por qué esa salida **no** resuelve el caso de `motivo_ausencia`, y por qué tampoco
   resuelve el de `id_empleado`. Los dos quedan fuera, y la cardinalidad no es el motivo de
   ninguno de los dos.
"""),

    code("""
for columna in ("motivo_ausencia", "id_empleado"):
    print(f"{columna}: {datos[columna].nunique()} categorías distintas")

print("\\ncategorías de motivo_ausencia con 2 registros o menos:")
frecuencias = datos["motivo_ausencia"].value_counts()
print(frecuencias[frecuencias <= 2].to_string())
"""),

    respuesta(4),

    md(f"""
---

## Antes de cerrar

Vuelvan a la [guía del repaso]({GUIA}) y abran, ahora sí, los criterios de éxito y las respuestas
comentadas de las dos actividades. Comparen **el razonamiento**, no solo la cifra: en las dos hay un
paso donde la respuesta correcta no es la lectura más rápida del resultado.
"""),
]


def con_ids(celdas: list[dict]) -> list[dict]:
    """nbformat 4.5+ exige un id por celda. Deterministas: regenerar no ensucia el diff."""
    for i, celda in enumerate(celdas):
        celda["id"] = f"c{i:02d}"
    return celdas


def main() -> None:
    celdas = con_ids(CELDAS)
    nb = {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            # Único caso aceptado de notebook sin salidas: es una plantilla que el estudiante
            # completa, y sus celdas con `?????` no se pueden ejecutar tal como están.
            "curso_plantilla": True,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    codigo = [c for c in celdas if c["cell_type"] == "code"]
    fuente = "".join("".join(c["source"]) for c in codigo)
    # Un andamiaje sin espacios que completar no es un andamiaje: si un `?????` se pierde en una
    # edición, la actividad queda resuelta y nadie lo nota al mirar el notebook.
    assert fuente.count("?????") == 12, f"esperaba 12 espacios `?????`, hay {fuente.count('?????')}"
    assert "curso_contrato" not in nb["metadata"]
    print(f"{SALIDA.relative_to(REPO_ROOT)}: {len(celdas)} celdas "
          f"({len(celdas) - len(codigo)} markdown / {len(codigo)} código), "
          f"{fuente.count('?????')} espacios por completar")


if __name__ == "__main__":
    main()
