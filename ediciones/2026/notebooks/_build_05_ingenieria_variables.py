#!/usr/bin/env python3
"""Construye notebooks/05-ingenieria-variables.ipynb.

POR QUÉ EXISTE UN GENERADOR: el `.ipynb` no se edita a mano. Si hay que corregir algo, se corrige acá y
se vuelve a ejecutar. En la clase 1 se perdieron dos correcciones por parchear el artefacto.

ESTRUCTURA: los tres tiempos de la `practica` del temario. (1) El docente arma el preprocesador mínimo a
nivel FACTURA y lo ejecuta. (2) Entre todos se vuelve a la tabla de LÍNEAS para ver dónde el one-hot deja
de servir. (3) Cada uno crea UNA variable nueva, la incorpora y la mide.

ESTA CLASE ESTRENA `metadata.curso_contrato` en la edición (decisión registrada en `HANDOFF.md`). El
contrato NO se escribe a mano: se deriva de la lista final de celdas al final de este archivo, así que
agregar o mover una celda lo actualiza solo. Ver `curso_checks` en `CLAUDE.md`.

CÓMO SE MIDE, y es el contrato de la clase entera: se ajusta un modelo lineal sobre `log1p` del monto, se
destransforma con `expm1` y se reporta la MEDIANA DEL ERROR ABSOLUTO EN LIBRAS. SIN R²: el curso
introduce las métricas de regresión en la clase 8 y las clases 1 a 4 no nombran ninguna. El motivo
completo está en la `calibracion` del temario y en `assets/ilustraciones/_build_c05_figuras.py`.

DOS UNIDADES DE OBSERVACIÓN, y el notebook las nombra cada vez que cambia: 522.504 LÍNEAS de producto
—un registro es un producto dentro de una factura— y 19.773 FACTURAS. El filtro que fija la base se declara
UNA vez y todas las cifras salen de ese estado.

FRONTERAS, heredadas del temario y verificadas al escribir:
  · No se parte en entrenamiento y prueba, no se nombra ningún mecanismo de fuga y no se compara un
    modelo contra otro: eso es la clase 6. La regla operativa se enuncia y se aplica.
  · No se discretiza: la teoría lo enseña, pero la práctica no lo repite.
  · El contrapunto de la variable construida desde el objetivo lo pone el docente en el CIERRE, no el
    notebook. Si estuviera acá, la etapa 3 se quedaría sin tiempo y el estudiante se llevaría la
    conclusión antes de haber hecho el trabajo.

NO ES UN INSTRUMENTO DE EVALUACIÓN: la Unidad 1 se acredita solo con la Solemne 1 (`programa.qmd`). El
material del estudiante usa actividad, criterio de éxito y evidencia observable.

CIFRAS: ninguna magnitud se escribe estática en el texto. Todas se calculan y se imprimen con `print`,
porque cada paso mueve la base del siguiente. NO se usa el visor de IPython para dar formato: era
andamiaje para poner negritas en la salida, y el estudiante escribe `print`.

Uso:  python ediciones/2026/notebooks/_build_05_ingenieria_variables.py
      (después: `make nb C=05` para validarlo, y `make nb C=05 KEEP_OUTPUT=...` para dejarlo con salidas)
"""
from __future__ import annotations

import base64
import json
import statistics
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "05-ingenieria-variables.ipynb"
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
<p>Semana 05: Ingeniería de variables y preprocesamiento</p>
</div>
</div>
"""),
    md("""
# 05 · Ingeniería de variables y preprocesamiento

La clase pasada **corregimos** lo que estaba mal en el archivo. Hoy **transformamos** lo que ya está
bien, para que el modelo pueda aprovecharlo.

El archivo es el mismo, con las siete decisiones de la clase 4 ya aplicadas.

## Qué vamos a hacer

1. Construir **un preprocesador** que trate por separado las columnas numéricas y las categóricas.
2. Ver **dónde el one-hot deja de servir**, con el código de producto.
3. **Crear una variable nueva** justificada desde el negocio, incorporarla y medirla.

## Criterio de éxito de la actividad

La justificación de la variable que creen explica **de dónde sale su valor** y **a qué hecho del negocio
corresponde**, sin apelar a cuánto bajó el error.
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

def num(x, dec=0):
    \"\"\"Miles con punto y decimales con coma, como en el resto del material del curso.\"\"\"
    return f'{x:,.{dec}f}'.replace(',', '@').replace('.', ',').replace('@', '.')
"""),
    md("""
## Cargar datos

El archivo viene en **Parquet**, que guarda el tipo de cada columna, así que no hay que adivinar nada al
leerlo. Ya usamos `read_parquet` en la clase 4.
"""),
    code("""
url_datasets = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
ruta_datos = '../datasets/' if os.path.exists('../datasets') else url_datasets
ventas = pd.read_parquet(ruta_datos + 'ventas_online_limpio.parquet')
"""),
    code("""
print(f'El archivo trae {num(len(ventas))} registros y {ventas.shape[1]} columnas.')
"""),
    md("""
Cada **registro** es una **línea de factura**: un producto dentro de una factura. No es una venta completa y
no es un cliente.

### El filtro que fija la base, y se declara una sola vez

Nos quedamos con las líneas que son **ventas de producto efectivas**: que sean producto, que no estén
canceladas, y que tengan cantidad y precio positivos. Todas las cifras de este cuaderno salen de ahí.

<!-- contrato: accion=fijar-base -->
"""),
    code("""
lineas = ventas[
    ventas.es_producto & ~ventas.cancelada & (ventas.cantidad > 0) & (ventas.precio_unitario > 0)
].copy()
lineas['monto_linea'] = lineas.cantidad * lineas.precio_unitario   # lo que vale esa línea
"""),
    code("""
print(f'Quedan {num(len(lineas))} líneas de las {num(len(ventas))} originales.')
"""),
    md("""
## Y ahora cada registro va a ser una factura

Para lo que sigue necesitamos **un registro por factura**, no por línea. O sea que cambiamos la **unidad de
observación**, igual que en la clase 2, y por eso lo decimos en voz alta.

### `agg`: resumir cada grupo con la función que corresponda

`groupby` ya lo conocen: separa las filas en grupos. `agg` es lo que va después: recibe, por cada columna
nueva, **de dónde sale y cómo se resume**. El monto de una factura es la **suma** de sus líneas, su
número de líneas es un **conteo**, y su país es el **primero** —todas sus líneas comparten país—.

El **mes** no viene como columna: se saca de la fecha con el accesor `.dt`, que da acceso a las partes
de una fecha —año, mes, día, hora—. Es la familia de variables derivadas «componentes de una fecha».

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
"""),
    code("""
facturas['mes'] = facturas.fecha.dt.month   # el mes sale de la fecha, no viene como columna
"""),
    code("""
print(f'De {num(len(lineas))} líneas quedaron {num(len(facturas))} facturas.')
"""),
    md(figura_base64(
        "c05-de-linea-a-factura.svg",
        "A la izquierda, cinco registros del archivo con el mismo número de documento repetido: son cinco "
        "líneas de una misma factura. A la derecha, esos cinco registros agrupados en uno solo, con su "
        "número de líneas, sus unidades y su monto total. Al pie, las líneas de producto se convierten "
        "en facturas.",
        width=980)),
    md("""
## Cómo vamos a medir

Necesitamos una forma de comparar dos maneras de escribir las mismas variables. Vamos a usar un **modelo
lineal** —el que ya conocen— como **instrumento de medición**, no como predictor: explica el monto de una
factura desde su número de líneas, sus unidades, su país y su mes.

Y lo vamos a reportar en **libras por factura**: la mitad de las facturas queda estimada con un error
menor a esa cifra. Nada de porcentajes abstractos.

### Dos detalles de la receta, y los dos importan

- El modelo trabaja sobre el **logaritmo** del monto, porque su distribución tiene una cola larguísima.
  Así que la predicción hay que **destransformarla** con `expm1` antes de compararla con libras reales.
- `log1p` calcula el logaritmo de **1 + x**, no de x. El logaritmo de cero no existe, y este archivo
  tiene facturas de una sola unidad.

### Dos nombres nuevos, y uno ya lo vieron en las slides

`make_pipeline` encadena pasos: le entregamos el preprocesador y el modelo, y devuelve un solo objeto
que aplica el primero y después el segundo. Es la forma corta de armar el **pipeline** del que hablamos
en clase.

Y una advertencia sobre la cifra: **se calcula sobre las mismas facturas con las que se ajustó el
modelo**. Sirve para comparar dos formas de escribir las mismas variables, que es lo único que
necesitamos hoy. Por qué eso no alcanza para saber si un modelo sirve es la clase 6.
"""),
    code("""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def error_libras(preprocesador, columnas):
    \"\"\"Mediana del error absoluto, en libras por factura.\"\"\"
    modelo = make_pipeline(preprocesador, LinearRegression())
    modelo.fit(facturas[columnas], np.log1p(facturas.monto_total))
    prediccion = np.expm1(modelo.predict(facturas[columnas]))
    return float(np.median(np.abs(prediccion - facturas.monto_total)))
"""),
    code("""
print('Referencia de lectura: la factura mediana vale '
      f'{num(facturas.monto_total.median())} libras.')
"""),
    md("""
# Primer tiempo · el preprocesador mínimo

Cuatro columnas entran: dos numéricas y dos categóricas. Cada tipo necesita algo distinto.

### Las tres piezas que hacen falta

- **`OneHotEncoder`** convierte una columna de categorías en una columna por categoría. Con
  `handle_unknown='ignore'` no se cae si mañana aparece un país que no estaba.
- **`FunctionTransformer`** aplica una función cualquiera a las columnas que se le indiquen; acá le
  vamos a pasar `log1p`.
- **`ColumnTransformer`** es el que reparte: recibe una lista de tripletas —nombre, transformador,
  columnas— y reúne todas las salidas en una sola matriz.

### Y por qué las numéricas pasan por el logaritmo

Porque es la decisión que la teoría de hoy justificó, y conviene no ejecutarla a ciegas: `unidades`
tiene una cola larguísima, y su máximo son **80.995 unidades en una sola factura** — la misma que la
clase 4 discutió y decidió **conservar**, porque era una venta real con su devolución.

Transformar la variable es lo que desarma esa cola **sin borrar la venta**. Acá no lo volvemos a
demostrar: se aplica la decisión y se sigue.
"""),
    code("""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

columnas_numericas = ['n_lineas', 'unidades']
columnas_categoricas = ['pais', 'mes']
"""),
    code("""
preprocesador = ColumnTransformer([
    ('num', FunctionTransformer(np.log1p), columnas_numericas),
    ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas),
])
"""),
    md("""
Hasta acá solo lo **declaramos**. Ahora lo ejecutamos sobre el archivo y miramos qué sale.

<!-- contrato: cifra=52 -->
"""),
    code("""
matriz = preprocesador.fit_transform(facturas[columnas_numericas + columnas_categoricas])
print(f'Entraron {len(columnas_numericas + columnas_categoricas)} columnas y salieron {matriz.shape[1]}.')
"""),
    md("""
### De dónde salen esas columnas

Las dos numéricas siguen siendo dos. Las categóricas son las que se multiplicaron: una columna por cada
valor distinto.
"""),
    code("""
for columna in columnas_categoricas:
    print(f'{columna:8s} {facturas[columna].nunique():>3} valores distintos')
"""),
    md("""
### Y casi todo lo que se agregó son ceros
"""),
    code("""
# la matriz viene comprimida, sin guardar los ceros; la expandimos solo para contarlos
densa = matriz.toarray() if hasattr(matriz, 'toarray') else matriz
print(f'El {num(100 * (densa == 0).mean())} % de esa matriz son ceros.')
"""),
    md(figura_base64(
        "c05-explosion-columnas.svg",
        "La matriz que recibe el modelo: cuatro facturas con sus dos columnas numéricas en logaritmo y "
        "después las columnas de país y de mes, donde cada fila tiene un único 1 y el resto son ceros. "
        "Al costado, el conteo de cuatro columnas a cincuenta y dos y el aviso de que la mayor parte de "
        "la matriz son ceros.",
        width=980)),
    md("""
### El error de este preprocesador

Guardemos la cifra, porque es la vara contra la que vamos a comparar todo lo demás.
"""),
    code("""
error_base = error_libras(preprocesador, columnas_numericas + columnas_categoricas)
print(f'Error mediano: {num(error_base, 1)} libras por factura.')
"""),
    md("""
# Segundo tiempo · dónde el one-hot deja de servir

Probemos con el **código de producto**. Y para eso hay que **volver a la tabla de líneas**, porque el
código de producto solo existe ahí: una factura tiene muchos productos, no un código.
"""),
    code("""
n_productos = lineas.codigo_producto.nunique()
print(f'Hay {num(n_productos)} productos distintos en {num(len(lineas))} líneas.')
"""),
    md("""
### Antes de ejecutar nada, hagamos la cuenta

Si le aplicáramos el one-hot, tendríamos una columna por producto y un registro por línea.
"""),
    code("""
celdas = len(lineas) * n_productos
gigas = celdas * 8 / 1e9   # 8 bytes por número en punto flotante
print(f'Serían {num(celdas / 1e6)} millones de celdas, '
      f'o {num(gigas, 1)} GB si se guardaran todas.')
"""),
    code("""
print(f'Y solo el {num(100 * len(lineas) / celdas, 3)} % sería distinto de cero.')
"""),
    md("""
### Qué concluimos

Dos cosas, y las dos importan:

1. Acá **la herramienta correcta no es esta**. Hay formas de guardar solo las celdas que no son cero, y
   con eso el archivo entra en memoria — pero el problema no era el espacio: el modelo seguiría teniendo
   que estimar un número por cada producto.
2. El código de producto **no puede entrar como una columna más** de la tabla por factura. Por eso hubo
   que agregar al principio. La agregación no fue una preferencia: fue la consecuencia de esto.
"""),
    md("""
# Tercer tiempo · ahora ustedes

Toca **crear una variable nueva** que el archivo no trae, incorporarla al preprocesador y medirla.

## La candidata: el precio de catálogo medio de la canasta

La idea de negocio es simple: **una factura de productos caros vale más que una de productos baratos**,
aunque las dos tengan la misma cantidad de líneas y de unidades. Eso el archivo no lo dice directamente,
pero se puede construir.

Dos pasos:

1. Para cada producto, su **precio habitual**: la mediana de todos los precios a los que se vendió. Sale
   con el mismo `agg` de antes, solo que agrupando por producto en vez de por factura.
2. Para cada factura, el **promedio de esos precios habituales** entre los productos que lleva.

### `merge`: pegarle a cada línea un dato que vive en otra tabla

`merge` une dos tablas por una columna que comparten. Acá la tabla chica es el catálogo de precios
habituales, y la columna compartida es `codigo_producto`.
"""),
    code("""
catalogo = lineas.groupby('codigo_producto').agg(
    precio_catalogo=('precio_unitario', 'median'),
).reset_index()
"""),
    code("""
catalogo.head()
"""),
    md("""
Ahora se lo pegamos a cada línea, y de ahí lo resumimos por factura.

<!-- contrato: accion=pegar-catalogo -->
"""),
    code("""
lineas_con_catalogo = lineas.merge(catalogo, on='codigo_producto', how='left')
"""),
    code("""
por_factura = lineas_con_catalogo.groupby('n_documento').precio_catalogo.mean()
facturas['precio_catalogo_medio'] = facturas.n_documento.map(por_factura)
"""),
    code("""
print(f'La variable nueva va de {num(facturas.precio_catalogo_medio.min(), 2)} '
      f'a {num(facturas.precio_catalogo_medio.max())} libras, '
      f'con mediana {num(facturas.precio_catalogo_medio.median(), 2)}.')
"""),
    md("""
## Incorporarla al preprocesador

Va por la **ruta numérica**, junto a las otras dos, así que también pasa por el logaritmo.
"""),
    code("""
columnas_numericas_con_precio = columnas_numericas + ['precio_catalogo_medio']
preprocesador_nuevo = ColumnTransformer([
    ('num', FunctionTransformer(np.log1p), columnas_numericas_con_precio),
    ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas),
])
"""),
    code("""
error_nuevo = error_libras(preprocesador_nuevo, columnas_numericas_con_precio + columnas_categoricas)
print(f'Error mediano: {num(error_nuevo, 1)} libras, contra {num(error_base, 1)} de antes.')
"""),
    md("""
## Ahora la parte que se mira

La cifra bajó, y eso está bien. Pero **no es lo que se les pide justificar**.

Escriban, en una o dos frases:

- **de dónde sale** el valor de esta variable, o sea qué cuenta exactamente;
- **a qué hecho del negocio** corresponde, o sea qué decisión o comportamiento real refleja.

Sin mencionar cuánto bajó el error.
"""),
    code("""
# escriban su justificación acá, entre las comillas
justificacion = \"\"\"

\"\"\"
print(justificacion.strip() or 'Todavía sin escribir.')
"""),
    md("""
## Y si les queda tiempo

Prueben **otra** variable propia y compárenla. Dos ideas del archivo, las dos defendibles desde el
negocio:

- las **unidades por línea** de la factura, que distingue una compra al detalle de una al por mayor;
- la **hora** de la factura, que sale de `fecha_factura` igual que el mes.

La pregunta es la misma: qué cuenta, y qué hecho del negocio refleja.
"""),
    md("""
## Atribución de datos

- **Creador:** Chen, D. (2012)
- **Fuente:** [UCI Machine Learning Repository — *Online Retail II*, dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** obra derivada en dos pasos. Las ocho columnas originales se renombraron al español
  en `snake_case`, sin traducir los valores; después se aplicaron las siete decisiones de limpieza de la
  clase 4, que eliminan filas repetidas y agregan seis columnas.
"""),
]


def contrato(celdas: list[dict]) -> dict:
    """Deriva `metadata.curso_contrato` de la lista FINAL de celdas.

    Se calcula, no se escribe a mano: agregar, quitar o mover una celda lo actualiza solo. Es la regla
    de `curso_checks` en `CLAUDE.md`, y existe porque un contrato con conteos de una versión anterior es
    peor que no tener contrato: pasa el chequeo y describe otro notebook.
    """
    codigo = [c for c in celdas if c["cell_type"] == "code"]
    fuente = "".join("".join(c["source"]) for c in codigo)
    lineas_por_celda = [
        len([l for l in "".join(c["source"]).splitlines() if l.strip() and not l.strip().startswith("#")])
        for c in codigo
    ]
    mediana = statistics.median(lineas_por_celda)
    return {
        "forma": {
            "markdown": sum(1 for c in celdas if c["cell_type"] == "markdown"),
            "codigo": len(codigo),
            "llamadas": [
                {"funcion": "error_libras", "argumentos": 2, "cantidad": fuente.count("error_libras(") - 1},
                {"funcion": "print", "argumentos": 1, "cantidad": fuente.count("print(")},
            ],
        },
        "acciones": [
            {"marcador": "fijar-base", "asignacion": "lineas"},
            {"marcador": "agregar-por-factura", "codigo": "groupby"},
            {"marcador": "pegar-catalogo", "codigo": "merge"},
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
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
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
    print(f"  contrato: mediana de líneas {nb['metadata']['curso_contrato']['max_mediana_lineas_codigo'] - 1}"
          f" · llamadas {forma['llamadas']}")


if __name__ == "__main__":
    main()
