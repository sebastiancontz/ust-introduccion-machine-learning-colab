#!/usr/bin/env python3
"""Construye los dos notebooks del repaso de la Unidad 2 (clases 5 a 8):

- `repaso-solemne-02.ipynb`: las tres actividades, con espacios `?????` para completar.
- `repaso-solemne-02-solucionario.ipynb`: las mismas celdas con el código resuelto y, en lugar de
  cada celda de respuesta, la respuesta comentada de ese paso.

POR QUÉ LA PLANTILLA NO LLEVA SALIDAS: sus celdas de código traen espacios `?????` que el
estudiante completa, así que no se pueden ejecutar tal como están. Por eso declara
`metadata.curso_plantilla: true`, el único caso en que el repo acepta un notebook sin salidas.

EL SOLUCIONARIO SÍ SE PUBLICA EJECUTADO, y las cifras de sus respuestas comentadas tienen que
coincidir con las salidas que muestra. Este script lo escribe sin salidas; se ejecuta después:

    python ediciones/2026/notebooks/_build_repaso_solemne_02.py
    python scripts/run_notebook_sandbox.py ediciones/2026/notebooks/repaso-solemne-02-solucionario.ipynb \\
        --keep-output ediciones/2026/notebooks/repaso-solemne-02-solucionario.ipynb

Si una cifra de RESPUESTAS no aparece en la salida ejecutada, la que manda es la salida: corregir
el texto, nunca forzar el dato.

LE HABLA AL ESTUDIANTE EN SINGULAR (docente, 2026-10-29): es la excepción a la regla general del
curso, que le habla al curso en plural. Rige para la guía y los dos notebooks de este repaso.

POR QUÉ LOS ARCHIVOS NO SON LOS DE LA EVALUACIÓN: la práctica de la Solemne 2 trabaja sobre otros
tres datasets. Si el repaso usara los mismos, practicar se convertiría en memorizar sus cifras.
Los tres de acá son sintéticos y los genera `datasets/_prep_repaso_solemne_02.py`.

Y ESO NO SE DICE EN EL MATERIAL VISIBLE (docente, 2026-09-10, para la Solemne 1): el material
describe cada actividad por lo que hay que hacer en ella, nunca por su parecido con la evaluación.
Esta nota es interna; no la promuevas a una celda.

APIs: todo lo que usan las tres actividades viene de las clases 5 a 8.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "repaso-solemne-02.ipynb"
SOLUCIONARIO = AQUI / "repaso-solemne-02-solucionario.ipynb"

LOGO = "data:image/svg+xml;base64," + base64.b64encode(
    (REPO_ROOT / "assets" / "logo-ust.svg").read_bytes()).decode("ascii")  # embebido: Colab no muestra el SVG externo
GUIA = ("https://sebastiancontz.github.io/ust-introduccion-machine-learning"
        "/ediciones/2026/guias/repaso-solemne-02.html")
COLAB = ("https://colab.research.google.com/github/sebastiancontz/"
         "ust-introduccion-machine-learning-colab/blob/main/ediciones/2026/notebooks/")
DATASETS = ("https://raw.githubusercontent.com/sebastiancontz/"
            "ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/")

# lo que completa cada `?????`, en orden de aparición
SOLUCION = [
    # actividad 1
    '"ambientes", "metros_cuadrados", "pisos_sin_ascensor", "distancia_km"',
    '"region", "tipo_vivienda", "incluye_embalaje", "objetos_especiales"',
    '"median"', "LinearRegression", '"median"', "X_train", "y_train",
    "flujo",
    # actividad 2
    '"most_frequent"', "LogisticRegression", "predict_proba", "umbral",
    "COSTO_FALSA_ALARMA", "COSTO_INASISTENCIA", "0.20",
    # actividad 3
    "profundidad", "pliegues", '"test_score"',
    "bosque", '"roc_auc"', "X_trainval", "y_trainval",
    'Pipeline([("preparar", preparar), ("modelo", DecisionTreeClassifier('
    "max_depth=3, random_state=SEMILLA))]).fit(X_trainval, y_trainval)",
]

# La respuesta comentada de cada paso, que en el solucionario reemplaza a su celda de respuesta.
# Las cifras salen de la ejecución del solucionario: ver el docstring.
RESPUESTAS = {
    "a1p1": """
**No pueden entrar cuatro columnas:**

- `id_servicio`: es un **identificador**; no describe la mudanza.
- `cajas_utilizadas`: se cuentan **al terminar** el servicio. Al cotizar no existen.
- `reclamo_posterior`: ocurre **después** del servicio.
- `monto_cobrado`: se **calcula a partir de las horas** que tomó el servicio. Es el objetivo dicho
  con otras palabras: su correlación con `horas_servicio` es 0,99.

**Los vacíos de `metros_cuadrados`** son 152 de 1.500, un 10 %: clientes que no informaron el
metraje al cotizar. Es información que también faltará en las cotizaciones nuevas, así que se
rellena dentro del flujo y no se descartan esos servicios.
""",
    "a1p2": """
| | Entrenamiento | Validación |
|:--|:--|:--|
| Piso (mediana) | 1,94 h | 1,90 h |
| Regresión lineal | 1,20 h | 1,11 h |

El modelo le gana al piso por **0,79 horas** en validación: de 1,90 a 1,11 horas de promedio del
error absoluto por servicio. Que la cifra de validación sea algo mejor que la de entrenamiento no es
una señal de alarma por sí sola: la diferencia es chica y puede deberse a la suerte de la partición.

**El piso es la mediana** porque la métrica es el error absoluto, y la constante que lo minimiza es
la mediana. El promedio minimiza otra cosa —el error cuadrático—, así que como piso de esta métrica
quedaría peor parado de lo que corresponde e inflaría la ventaja del modelo.
""",
    "a1p3": """
La colega A obtiene **1,11 horas** en validación y el colega B, **0,13 horas**.

- **Colega A: fuga de procedimiento.** Calcula la mediana que rellena los vacíos con el archivo
  completo, que incluye validación y prueba. Su cifra es la misma que la tuya porque esa mediana
  casi no cambia —78 m² con todo el archivo y 77 m² solo con entrenamiento—. Se corrige igual: la
  regla no se respeta porque cambie la cifra, sino porque el error **no se nota**, y en otro archivo
  o con otra transformación sí la cambiaría. Este atajo lo **evita un `Pipeline`**, que ajusta la
  imputación solo con entrenamiento.
- **Colega B: fuga de origen.** Su flujo está bien construido, pero `monto_cobrado` se calcula con
  las horas del servicio y `cajas_utilizadas` se cuenta al terminar. Ninguna de las dos existe al
  cotizar. Una cifra diez veces mejor no es un mejor modelo: es la respuesta colándose por dos
  columnas. Un `Pipeline` **no lo evita**, porque el problema no está en cuándo se calcula cada cosa
  sino en de dónde salen los datos.
""",
    "a1p4": """
En prueba, el promedio del error absoluto es de **1,25 horas** por servicio. Un mensaje posible:

> Con lo que sabemos al cotizar, el modelo estima la duración de cada mudanza con un error promedio
> de una hora y cuarto. La regla simple de asignar a todas la duración típica se equivoca en casi
> dos horas. Es un promedio: algunas mudanzas se desviarán bastante más, así que conviene dejar
> holgura en la última del día. El modelo no explica por qué una mudanza tarda más.

**No se vuelve a ajustar después de mirar prueba** porque, en cuanto se usa para decidir, la cifra
de prueba deja de estimar lo que pasará con servicios nuevos y queda optimista. Que prueba salga
algo peor que validación es esperable: ninguna de las dos es la cifra «verdadera».
""",
    "a2p1": """
El piso responde siempre «se presenta». Su accuracy en validación es **0,826** y detecta **0 de 87**
inasistencias.

Para la administración: acierta el 83 % porque la mayoría de los pacientes sí llega, no porque sepa
algo. Es exactamente la regla de no llamar a nadie, y no anticipa ni una hora vacía.
""",
    "a2p3": """
**Tres umbrales sobre validación**, leídos del barrido del paso 2:

| Criterio | Umbral | Costo | A llamar |
|:--|:--|:--|:--|
| Mínimo costo | 0,20 | $1.236.000 | 32,4 % |
| Máximo F1 (0,389) | 0,30 | $1.292.000 | 12,4 % |
| 0,5 de `predict` | 0,50 | $1.652.000 | 1,6 % |

Se elige **0,20**. El umbral de F1 cuesta $56.000 más, porque F1 pesa igual una falsa alarma y una
inasistencia escapada, y acá la segunda cuesta cinco veces la primera. El de 0,5 cuesta $416.000 más:
casi no llama a nadie.

Un matiz que vale la pena notar: 0,15 cuesta $1.244.000, apenas $8.000 más que 0,20, pero obliga a
llamar al 48 % de las citas. Con costos tan parecidos, la diferencia no distingue a los dos umbrales,
y 0,20 tiene la ventaja de pedirle a recepción la mitad de las llamadas.

**En prueba**, con 0,20: precision **0,338**, recall **0,598** y proporción a llamar **0,308**. Un
mensaje posible:

> Recepción llamaría a unas 31 de cada 100 citas, y con eso anticiparía unas 6 de cada 10
> inasistencias. De cada tres llamadas, una es a un paciente que efectivamente iba a faltar; las otras
> dos son a pacientes que sí iban a venir, y es un costo que conviene pagar. No usamos el corte de
> 0,5 porque casi no llamaría a nadie. La probabilidad ordena a quién llamar primero, pero no
> asegura que un paciente con 0,6 vaya a faltar, ni que 6 de cada 10 de esos pacientes falten.
""",
    "a3p1": """
La curva sube hasta la profundidad 3 y después baja fuera de muestra, mientras en entrenamiento
sigue subiendo hasta 1,00 sin límite de profundidad.

Se elige **profundidad 3**, con **0,738 ± 0,037**. No la de mejor entrenamiento, porque esa premia
memorizar: sin límite, el árbol acierta todo lo que vio y ordena peor que uno de dos niveles.

Las profundidades 2, 3 y 4 difieren en menos de 0,015, bastante menos que su desviación de casi
0,04: **no se distinguen**. Elegir 2 también es defendible, porque da reglas más simples a casi el
mismo desempeño.
""",
    "a3p2": """
La mejor combinación es 300 árboles, `min_samples_leaf` = 20 y `max_features` = `"sqrt"`, con
**0,750 ± 0,042**.

La búsqueda hace 2 × 2 × 2 = 8 combinaciones, por 5 pliegues, más el reajuste del ganador:
**41 entrenamientos**.

Subir `min_samples_leaf` de 1 a 20 mueve la cifra de alrededor de 0,71 a alrededor de 0,75. Pasar
de 100 a 300 árboles la mueve menos de 0,01. La complejidad que importa en un bosque es la de
**cada árbol**: más árboles no lo sobreajustan, solo lo estabilizan.
""",
    "a3p3": """
El mejor bosque, 0,750 ± 0,042, supera al mejor árbol, 0,738 ± 0,037, por 0,012. Esa diferencia es
**menor que la dispersión entre pliegues**, así que no distingue a los dos modelos. Con un empate,
decide lo que se pierde al cambiar: el árbol se explica con reglas, que es lo que pidió la gerenta.
Se recomienda el **árbol de profundidad 3**, reentrenado con todo `X_trainval`.

Su ROC-AUC en prueba es **0,710**. Un mensaje posible:

> Recomendamos un árbol de tres niveles de preguntas: ordena las cotizaciones casi igual que un modelo
> más complejo, y se puede explicar al equipo de ventas con reglas. Su cifra dice qué tan bien pone
> arriba las cotizaciones que se concretan, no cuántas se van a concretar. Le informamos los ocho
> intentos del modelo complejo, y no solo el mejor, para que vea que la ventaja del ganador es menor
> que lo que varía la cifra de una partición a otra.

Si eliges el bosque, la decisión es aceptable siempre que reconozcas que su ventaja no supera la
dispersión y que pesa contra la preferencia declarada. Su ROC-AUC en prueba es 0,706.
""",
}


def md(texto: str, **metadata) -> dict:
    return {"cell_type": "markdown", "metadata": metadata,
            "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def respuesta(clave: str, paso: int, campos: str) -> dict:
    """Celda de respuesta con campos; en el solucionario la reemplaza RESPUESTAS[clave]."""
    return md(f"### Respuesta — paso {paso}\n\n" + campos.strip("\n"), repaso_respuesta=clave)


def carga(archivo: str, variable: str) -> str:
    return (f'RUTA = (\n    "{DATASETS}"\n    "{archivo}"\n)\n'
            f"{variable} = pd.read_csv(RUTA)\nSEMILLA = 42\n")


def encabezado(subtitulo: str) -> dict:
    return md(f"""
<div style="display:flex; align-items:center; gap:18px; text-align:left">
<img src="{LOGO}" width="100" alt="Logo de la Universidad Santo Tomás">
<div>
<p>Ingeniería en Información y Control de Gestión</p>
<p>Facultad de Economía y Negocios</p>
<p>Introducción a Machine Learning</p>
<p>{subtitulo}</p>
</div>
</div>
""")


PORTADA = [
    encabezado("Repaso de la Unidad 2: Algoritmos de Machine Learning"),

    md(f"""
# Repaso · Actividades de las clases 5 a 8

Tres actividades para practicar con datos lo que se trabajó entre las clases 5 y 8: un flujo sin
fuga para un problema de regresión, un umbral de clasificación decidido por costos, y la elección de
un modelo con validación cruzada. Cada paso se responde completando los espacios marcados `?????` o
escribiendo en las celdas de respuesta.

Los tres archivos son **sintéticos**, generados para este repaso: no representan a ninguna
organización real, y **no** son los que verás en la evaluación. Lo que hay que llevarse es el
procedimiento, no las cifras de un archivo.

**Cómo trabajar.** Haz una copia de este notebook en tu propia unidad y resuelve primero sin
consultar. Los **criterios de éxito** están en la [guía del repaso]({GUIA}), y el código resuelto
con la respuesta comentada de cada paso, en el
[solucionario]({COLAB}repaso-solemne-02-solucionario.ipynb). Ábrelo después de intentarlo.

Cada decisión va con su justificación escrita. Una cifra sin la lectura que la acompaña no muestra
el razonamiento que la unidad pide.
"""),
]

PORTADA_SOLUCIONARIO = [
    encabezado("Repaso de la Unidad 2: solucionario"),

    md(f"""
# Repaso · Solucionario de las actividades de las clases 5 a 8

Este notebook trae las tres actividades del repaso **resueltas y ejecutadas**: el código completo,
sus salidas y, después de cada paso, la respuesta comentada.

**Ábrelo después de intentarlo** en el [notebook de actividades]({COLAB}repaso-solemne-02.ipynb).
Leer la solución antes de resolver se siente como aprender, pero no ejercita lo que se necesita para
resolver un archivo que no conoces.

Al comparar, fíjate en **el razonamiento**, no solo en la cifra: en las tres actividades hay un paso
donde la respuesta correcta no es la lectura más rápida del resultado. Las respuestas escritas son
**una** forma de decirlo, no la única; los criterios de éxito de la [guía del repaso]({GUIA}) dicen
qué tiene que estar.
"""),
]

CELDAS = [
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas, así que la celda siguiente termina en segundos.
"""),

    code("""
%%capture
!pip install -q pandas numpy scikit-learn
"""),

    # ── Actividad 1 ───────────────────────────────────────────────────────────────────────────
    md("""
---

## Actividad 1 · Cuántas horas tomará la mudanza, sin mirar el futuro

**Cuatro pasos.** Clases 5 y 6.

Una empresa de mudanzas cotiza cada servicio por teléfono y hoy asigna camiones y cuadrillas «a ojo»:
algunos días sobran personas y otros la última mudanza termina de noche. La jefa de operaciones te
encarga un modelo de **regresión** que estime, **en el momento de cotizar**, cuántas horas tomará el
servicio.

El archivo `servicios_mudanza.csv` tiene 1.500 mudanzas ya realizadas. Cada registro es un servicio y
la variable objetivo es `horas_servicio`, en horas. La celda siguiente lo carga en `servicios`.
"""),

    code(f"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

{carga("servicios_mudanza.csv", "servicios")}
servicios.info()
"""),

    md("""
### Paso 1 · Qué sabe el modelo cuando se cotiza

1. Revisa las 13 columnas e indica cuáles **no** pueden entrar como predictoras.
2. Para cada una, di por qué: ¿es un identificador, se conoce recién al terminar la mudanza, o se
   calcula a partir de la propia variable objetivo?
3. `metros_cuadrados` tiene vacíos. Di qué representan, mirando cuándo se llena esa columna.
"""),

    respuesta("a1p1", 1, """
**1. Columnas que no pueden entrar, cada una con su motivo** (agrega una línea por columna)

- Columna: … · Motivo: …

**2. Qué representan los vacíos de `metros_cuadrados`:** …
"""),

    md("""
### Paso 2 · El flujo y su piso

La partición ya viene hecha: 60 % entrenamiento, 20 % validación y 20 % prueba.

1. Completa las dos listas de columnas con **tu** decisión del paso 1.
2. Completa el flujo: la ruta numérica imputa los vacíos con la mediana, la categórica codifica con
   `OneHotEncoder`, y el modelo es una regresión lineal. El piso predice siempre la mediana.
3. Entrena los dos **solo con entrenamiento** y reporta el promedio del error absoluto, en horas,
   de cada uno en entrenamiento y en validación.
4. ¿El modelo le gana al piso, y por cuánto? ¿Por qué el piso es la mediana y no el promedio, si la
   métrica es el error absoluto?
"""),

    code("""
# completa los espacios marcados abajo
NUMERICAS = [?????]
CATEGORICAS = [?????]

X = servicios[NUMERICAS + CATEGORICAS]
y = servicios["horas_servicio"]

# 60 % entrenamiento, 20 % validación y 20 % prueba
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEMILLA)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=SEMILLA)

preparar = ColumnTransformer([
    ("num", SimpleImputer(strategy=?????), NUMERICAS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
])
flujo = Pipeline([("preparar", preparar), ("modelo", ?????())])
piso = Pipeline([("preparar", clone(preparar)),
                 ("modelo", DummyRegressor(strategy=?????))])

for nombre, candidato in (("piso", piso), ("regresión lineal", flujo)):
    candidato.fit(?????, ?????)
    error_train = mean_absolute_error(y_train, candidato.predict(X_train))
    error_val = mean_absolute_error(y_val, candidato.predict(X_val))
    print(f"{nombre}: {error_train:.2f} h en entrenamiento, {error_val:.2f} h en validación")
"""),

    respuesta("a1p2", 2, """
**1. Promedio del error absoluto, en horas**

| | Entrenamiento | Validación |
|:--|:--|:--|
| Piso | … | … |
| Regresión lineal | … | … |

**2. ¿Le gana al piso, y por cuánto?** …

**3. Por qué el piso es la mediana:** …
"""),

    md("""
### Paso 3 · Dos colegas, dos atajos

Dos colegas del equipo armaron su propia versión. La celda ejecuta el código de los dos tal como lo
escribieron.

- **Colega A** rellena los vacíos de `metros_cuadrados` con la mediana del archivo completo y
  **después** parte los datos. Usa las mismas columnas que tú.
- **Colega B** usa un `Pipeline` igual al tuyo, ajustado solo con entrenamiento, pero agrega
  `cajas_utilizadas` y `monto_cobrado` porque «mejoran mucho el modelo».

1. Ejecuta la celda y reporta las cifras de validación de los dos colegas.
2. Para cada colega, di si comete fuga y de qué tipo: **de procedimiento** o **de origen**.
3. La cifra de la colega A es prácticamente igual a la tuya. ¿Por qué igual corresponde corregir su
   código? ¿Cuál de los dos atajos habría evitado un `Pipeline`, y cuál no?
"""),

    code("""
# los flujos de los colegas, tal como los escribieron: no los modifiques

# colega A: rellena con la mediana de TODO el archivo y después parte
copia = servicios.copy()
copia["metros_cuadrados"] = copia["metros_cuadrados"].fillna(copia["metros_cuadrados"].median())
Xa_trainval, _, ya_trainval, _ = train_test_split(
    copia[NUMERICAS + CATEGORICAS], y, test_size=0.2, random_state=SEMILLA)
Xa_train, Xa_val, ya_train, ya_val = train_test_split(
    Xa_trainval, ya_trainval, test_size=0.25, random_state=SEMILLA)
flujo_a = Pipeline([
    ("preparar", ColumnTransformer([
        ("num", "passthrough", NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])),
    ("modelo", LinearRegression()),
]).fit(Xa_train, ya_train)
print(f"colega A: {mean_absolute_error(ya_val, flujo_a.predict(Xa_val)):.2f} h en validación")

# colega B: un Pipeline correcto, con dos columnas más
NUM_B = NUMERICAS + ["cajas_utilizadas", "monto_cobrado"]
Xb_trainval, _, yb_trainval, _ = train_test_split(
    servicios[NUM_B + CATEGORICAS], y, test_size=0.2, random_state=SEMILLA)
Xb_train, Xb_val, yb_train, yb_val = train_test_split(
    Xb_trainval, yb_trainval, test_size=0.25, random_state=SEMILLA)
flujo_b = Pipeline([
    ("preparar", ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_B),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])),
    ("modelo", LinearRegression()),
]).fit(Xb_train, yb_train)
print(f"colega B: {mean_absolute_error(yb_val, flujo_b.predict(Xb_val)):.2f} h en validación")
"""),

    respuesta("a1p3", 3, """
**1. Cifras de validación:** colega A: … h · colega B: … h

**2. Diagnóstico**

- Colega A · ¿fuga? … · Tipo: … · Por qué: …
- Colega B · ¿fuga? … · Tipo: … · Por qué: …

**3. Por qué se corrige A aunque la cifra no cambie:** …

**4. Qué atajo evita un `Pipeline` y cuál no:** …
"""),

    md("""
### Paso 4 · La cifra para la jefa de operaciones

1. Con **tu** flujo del paso 2, calcula **una sola vez** el promedio del error absoluto, en horas,
   sobre prueba.
2. Redacta el mensaje para la jefa de operaciones, en no más de cinco líneas y sin jerga técnica:
   qué cifra le entregas, contra qué referencia se compara, y qué **no** le promete el modelo.
3. ¿Por qué no se vuelve a ajustar el flujo si la cifra de prueba sale peor que la de validación?
"""),

    code("""
# la prueba se abre UNA sola vez, con el flujo del paso 2
error_test = mean_absolute_error(y_test, ?????.predict(X_test))
print(f"prueba: {error_test:.2f} h")
"""),

    respuesta("a1p4", 4, """
**Promedio del error absoluto en prueba, en horas:** …

**Mensaje para la jefa de operaciones** (máximo cinco líneas, sin jerga técnica)

- Qué cifra le entregas y qué significa para asignar cuadrillas: …
- Contra qué referencia se compara: …
- Qué no le promete el modelo: …

**Por qué no se reajusta después de mirar prueba:** …
"""),

    # ── Actividad 2 ───────────────────────────────────────────────────────────────────────────
    md("""
---

## Actividad 2 · A qué pacientes llamar el día antes

**Tres pasos.** Clase 7.

Un centro odontológico pierde horas clínicas porque cerca de uno de cada seis pacientes no se
presenta a su cita. La recepción puede llamar el día anterior para confirmar, y si el paciente no
confirma, la hora se ofrece a la lista de espera. Te encargan un modelo de **clasificación** que
estime qué citas van a quedar vacías.

La administración te entrega dos cifras:

- Llamar a un paciente que **sí** iba a venir ocupa tiempo de recepción y molesta: cada falsa alarma
  cuesta **$4.000**.
- Una cita a la que el paciente no llega, sin que nadie lo haya anticipado, deja una hora clínica
  vacía: cuesta **$20.000**.

El archivo `inasistencias_citas.csv` tiene 2.500 citas; la variable objetivo es `no_se_presento`. La
celda lo carga en `citas`, codifica el objetivo con 1 = no se presentó, y deja hecha la partición
estratificada 60/20/20.
"""),

    code(f"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

{carga("inasistencias_citas.csv", "citas")}
COSTO_FALSA_ALARMA = 4_000    # pesos por paciente llamado que sí iba a venir
COSTO_INASISTENCIA = 20_000   # pesos por hora clínica que queda vacía sin aviso

NUMERICAS = ["dias_anticipacion", "edad_paciente", "citas_previas",
             "inasistencias_previas", "distancia_km"]
CATEGORICAS = ["tipo_atencion", "canal_agenda", "bloque_horario", "recordatorio_enviado"]
X = citas[NUMERICAS + CATEGORICAS]
y = (citas["no_se_presento"] == "si").astype(int)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=SEMILLA, stratify=y_trainval)
print(f"validación: {{len(y_val)}} citas, {{y_val.sum()}} inasistencias")
"""),

    md("""
### Paso 1 · El piso que acierta casi siempre

1. Completa el baseline que responde siempre la clase más frecuente y reporta su accuracy en
   validación y cuántas inasistencias detecta.
2. Explica, en dos o tres líneas para la administración, por qué esa accuracy no prueba que el piso
   sirva.
"""),

    code("""
# completa el espacio marcado abajo
piso = DummyClassifier(strategy=?????).fit(X_train, y_train)
lista_piso = piso.predict(X_val)
print(f"accuracy del piso: {accuracy_score(y_val, lista_piso):.3f}")
print(f"inasistencias detectadas: {lista_piso[y_val == 1].sum()} de {y_val.sum()}")
"""),

    respuesta("a2p1", 1, """
**1. Accuracy del piso en validación:** … · **Inasistencias detectadas:** … de …

**2. Por qué esa accuracy no prueba nada** (dos o tres líneas): …
"""),

    md("""
### Paso 2 · La probabilidad y el barrido de umbrales

1. Completa el flujo con una regresión logística, entrénalo y obtén la **probabilidad** de
   inasistencia de cada cita de validación, no la clase.
2. Completa el barrido: para cada umbral se cuentan falsas alarmas e inasistencias escapadas, y se
   calculan precision, recall, F1, la proporción de citas a llamar y el costo en pesos.

La lectura del barrido se responde en el paso 3.
"""),

    code("""
# completa los espacios marcados abajo
modelo = Pipeline([
    ("preparar", ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAS),
    ])),
    ("modelo", ?????(max_iter=2000)),
])
modelo.fit(X_train, y_train)
prob_val = modelo.?????(X_val)[:, 1]   # la probabilidad de inasistencia

GRILLA = np.round(np.arange(0.05, 0.61, 0.05), 2)
filas = []
for umbral in GRILLA:
    lista = np.where(prob_val >= ?????, 1, 0)
    _, falsas, escapadas, _ = confusion_matrix(y_val, lista, labels=[0, 1]).ravel()
    filas.append({"umbral": umbral, "falsas_alarmas": falsas,
                  "escapadas": escapadas,
                  "precision": precision_score(y_val, lista, zero_division=0),
                  "recall": recall_score(y_val, lista),
                  "f1": f1_score(y_val, lista, zero_division=0),
                  "a_llamar": lista.mean(),
                  "costo": ????? * falsas + ????? * escapadas})
barrido = pd.DataFrame(filas)
barrido.round(3)
"""),

    md("""
### Paso 3 · El umbral, decidido por lo que cuesta cada error

1. Identifica en el barrido tres umbrales: el que **minimiza el costo**, el que **maximiza F1** y
   el 0,5 que usa `predict`. Reporta el costo de cada uno.
2. Elige el umbral **sobre validación** y justifica por qué no es el de F1.
3. Aplica ese umbral **una sola vez** a prueba y reporta precision, recall y proporción a llamar.
4. Escribe, en no más de cinco líneas, el mensaje para la jefa de recepción: a cuántas citas de cada
   cien llamar, cuántas de cada diez inasistencias se anticipan, y qué **no** garantiza la
   probabilidad que entrega el modelo.
"""),

    code("""
# tu decisión, tomada sobre validación
UMBRAL = ?????

# la prueba se abre UNA sola vez
lista_test = np.where(modelo.predict_proba(X_test)[:, 1] >= UMBRAL, 1, 0)
print(f"prueba: precision {precision_score(y_test, lista_test):.3f}, "
      f"recall {recall_score(y_test, lista_test):.3f}, "
      f"a llamar {lista_test.mean():.3f}")
"""),

    respuesta("a2p3", 3, """
**1. Tres umbrales sobre validación**

| Criterio | Umbral | Costo |
|:--|:--|:--|
| Mínimo costo | … | … |
| Máximo F1 | … | … |
| 0,5 de `predict` | 0,5 | … |

**2. Umbral elegido y por qué no el de F1:** …

**3. En prueba:** precision … · recall … · proporción a llamar …

**4. Mensaje para la jefa de recepción** (máximo cinco líneas)

- Cuántas citas de cada cien llamar: …
- Cuántas de cada diez inasistencias se anticipan: …
- Qué no garantiza la probabilidad: …
"""),

    # ── Actividad 3 ───────────────────────────────────────────────────────────────────────────
    md("""
---

## Actividad 3 · Qué cotizaciones seguir primero

**Tres pasos.** Clase 8.

Una distribuidora de insumos industriales envía cientos de cotizaciones al mes, y el equipo
comercial no alcanza a hacerles seguimiento a todas. La gerenta comercial quiere **ordenar** las
cotizaciones según su probabilidad de concretarse en venta, para llamar primero a las que más
prometen. Prefiere un modelo que se pueda explicar al equipo de ventas con reglas simples.

El archivo `cotizaciones_b2b.csv` tiene 2.000 cotizaciones; la variable objetivo es `se_concreto`.
La celda lo carga en `cotizaciones`, codifica el objetivo con 1 = se concretó, separa prueba (20 %,
estratificada) y deja definidos el preprocesamiento `preparar` y los cinco pliegues estratificados
`pliegues`. La métrica es ROC-AUC, porque lo que se necesita es **ordenar**.
"""),

    code(f"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

{carga("cotizaciones_b2b.csv", "cotizaciones")}
NUMERICAS = ["monto_cotizado", "descuento_pct", "dias_respuesta", "cantidad_productos"]
CATEGORICAS = ["rubro_cliente", "tamano_cliente", "es_cliente_recurrente", "canal_origen"]
X = cotizaciones[NUMERICAS + CATEGORICAS]
y = (cotizaciones["se_concreto"] == "si").astype(int)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEMILLA, stratify=y)
pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
preparar = ColumnTransformer([("num", "passthrough", NUMERICAS),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)])
print(f"X_trainval: {{len(X_trainval)}} cotizaciones · X_test: {{len(X_test)}} cotizaciones")
"""),

    md("""
### Paso 1 · La curva en U del árbol

1. Completa la celda que, para cada profundidad, estima con validación cruzada la ROC-AUC del árbol
   en entrenamiento y fuera de muestra, con la media y la desviación entre pliegues.
2. ¿Qué profundidad eliges, y por qué no la que tiene mejor desempeño en entrenamiento? Usa la
   desviación para decir si las mejores profundidades se distinguen entre sí.
"""),

    code("""
# completa los espacios marcados abajo
filas = []
for profundidad in [1, 2, 3, 4, 5, 6, 8, 12, None]:
    arbol = Pipeline([("preparar", preparar),
                      ("modelo", DecisionTreeClassifier(max_depth=?????,
                                                         random_state=SEMILLA))])
    res = cross_validate(arbol, X_trainval, y_trainval, cv=?????,
                         scoring="roc_auc", return_train_score=True)
    filas.append({"profundidad": str(profundidad),
                  "auc_entrenamiento": res["train_score"].mean(),
                  "auc_validacion": res[?????].mean(),
                  "desviacion": res["test_score"].std()})
curva = pd.DataFrame(filas)
curva.round(3)
"""),

    respuesta("a3p1", 1, """
**1. Profundidad elegida:** … · ROC-AUC fuera de muestra: … ± …

**2. Por qué no la mejor de entrenamiento:** …

**3. ¿Se distinguen las mejores profundidades entre sí?** …
"""),

    md("""
### Paso 2 · La búsqueda sobre el bosque

1. Completa la búsqueda con `GridSearchCV` sobre el pipeline del bosque y muestra la tabla de
   **todos** los intentos ordenada por ranking.
2. ¿Cuántos entrenamientos hace la búsqueda en total, contando el reajuste final? Muestra el
   cálculo.
3. ¿Qué cambia más la cifra: pasar de 100 a 300 árboles, o subir `min_samples_leaf` de 1 a 20?
   ¿Qué dice eso sobre dónde está la complejidad de un bosque?
"""),

    code("""
# completa los espacios marcados abajo
bosque = Pipeline([("preparar", preparar),
                   ("modelo", RandomForestClassifier(random_state=SEMILLA))])
grilla = {"modelo__n_estimators": [100, 300],
          "modelo__min_samples_leaf": [1, 20],
          "modelo__max_features": ["sqrt", 0.5]}
busqueda = GridSearchCV(?????, grilla, cv=pliegues, scoring=?????)
busqueda.fit(?????, ?????)

intentos = pd.DataFrame(busqueda.cv_results_)[[
    "param_modelo__n_estimators", "param_modelo__min_samples_leaf",
    "param_modelo__max_features", "mean_test_score", "std_test_score", "rank_test_score"]]
intentos.sort_values("rank_test_score").round(3)
"""),

    respuesta("a3p2", 2, """
**1. Mejor combinación:** … · ROC-AUC fuera de muestra: … ± …

**2. Entrenamientos en total** (con el cálculo): …

**3. Qué mueve más la cifra, y qué dice eso:** …
"""),

    md("""
### Paso 3 · Árbol o bosque, y la cifra final

1. Compara el mejor árbol del paso 1 con el mejor bosque del paso 2: ¿la diferencia supera la
   dispersión entre pliegues? Decide cuál recomiendas, considerando también la preferencia de la
   gerenta.
2. Con el modelo elegido, reentrenado con todo `X_trainval`, calcula **una sola vez** la ROC-AUC en
   prueba.
3. Escribe, en no más de cinco líneas, el mensaje para la gerenta comercial: qué modelo recomiendas
   y por qué, qué significa la cifra para ordenar las cotizaciones, y por qué le informas todos los
   intentos y no solo el ganador.
"""),

    code("""
# la prueba se abre UNA sola vez, con el modelo que elegiste
# GridSearchCV ya reajustó a su ganador con todo X_trainval: si eliges el bosque, usa busqueda
# si eliges el árbol, defínelo con su profundidad y entrénalo con todo X_trainval
elegido = ?????
auc_test = roc_auc_score(y_test, elegido.predict_proba(X_test)[:, 1])
print(f"ROC-AUC en prueba: {auc_test:.3f}")
"""),

    respuesta("a3p3", 3, """
**1. Comparación:** mejor árbol … ± … · mejor bosque … ± … · ¿la diferencia supera la dispersión? …

**2. Modelo recomendado y por qué:** …

**3. ROC-AUC en prueba:** …

**4. Mensaje para la gerenta comercial** (máximo cinco líneas)

- Qué modelo recomiendas y por qué: …
- Qué significa la cifra para ordenar las cotizaciones: …
- Por qué le informas todos los intentos: …
"""),
]

CIERRE = md(f"""
---

## Antes de cerrar

Abre el [solucionario]({COLAB}repaso-solemne-02-solucionario.ipynb) y compara **el razonamiento**,
no solo la cifra: en las tres actividades hay un paso donde la respuesta correcta no es la lectura
más rápida del resultado. Los criterios de éxito de cada actividad están en la
[guía del repaso]({GUIA}).
""")


def con_ids(celdas: list[dict]) -> list[dict]:
    """nbformat 4.5+ exige un id por celda. Deterministas: regenerar no ensucia el diff."""
    for i, celda in enumerate(celdas):
        celda["id"] = f"c{i:02d}"
    return celdas


def notebook(celdas: list[dict], plantilla: bool) -> dict:
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if plantilla:
        # Único caso aceptado de notebook sin salidas: es una plantilla que el estudiante
        # completa, y sus celdas con `?????` no se pueden ejecutar tal como están.
        metadata["curso_plantilla"] = True
    return {"cells": con_ids(celdas), "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def resolver(celdas: list[dict]) -> list[dict]:
    """Completa cada `?????` con SOLUCION y cambia cada celda de respuesta por la comentada."""
    resueltas = json.loads(json.dumps(celdas))
    pendientes = iter(SOLUCION)
    usadas = set()
    for celda in resueltas:
        clave = celda["metadata"].pop("repaso_respuesta", None)
        if clave:
            paso = "".join(celda["source"]).splitlines()[0].rsplit(" ", 1)[-1]
            texto = f"### Respuesta comentada — paso {paso}\n\n" + RESPUESTAS[clave].strip("\n")
            celda["source"] = texto.splitlines(keepends=True)
            usadas.add(clave)
        elif celda["cell_type"] == "code":
            fuente = "".join(celda["source"])
            while "?????" in fuente:
                fuente = fuente.replace("?????", next(pendientes), 1)
            celda["source"] = fuente.splitlines(keepends=True)
    assert next(pendientes, None) is None, "sobran soluciones: SOLUCION y los ????? no calzan"
    assert usadas == set(RESPUESTAS), f"respuestas sin celda: {set(RESPUESTAS) - usadas}"
    return resueltas


def escribir(ruta: Path, nb: dict) -> None:
    ruta.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    actividades = PORTADA + CELDAS + [CIERRE]
    codigo = "".join("".join(c["source"]) for c in actividades if c["cell_type"] == "code")
    # Un andamiaje sin espacios que completar no es un andamiaje: si un `?????` se pierde en una
    # edición, la actividad queda resuelta y nadie lo nota al mirar el notebook.
    assert codigo.count("?????") == len(SOLUCION), \
        f"esperaba {len(SOLUCION)} espacios `?????`, hay {codigo.count('?????')}"
    plantilla = json.loads(json.dumps(actividades))
    for celda in plantilla:
        celda["metadata"].pop("repaso_respuesta", None)
    escribir(SALIDA, notebook(plantilla, plantilla=True))

    solucionario = PORTADA_SOLUCIONARIO + resolver(CELDAS)
    escribir(SOLUCIONARIO, notebook(solucionario, plantilla=False))
    print(f"{SALIDA.relative_to(REPO_ROOT)}: {len(actividades)} celdas, "
          f"{codigo.count('?????')} espacios por completar")
    print(f"{SOLUCIONARIO.relative_to(REPO_ROOT)}: {len(solucionario)} celdas, SIN ejecutar")


if __name__ == "__main__":
    main()
