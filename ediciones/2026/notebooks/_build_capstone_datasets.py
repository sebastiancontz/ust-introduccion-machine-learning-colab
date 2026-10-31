"""Genera `capstone-datasets.ipynb`: los seis casos del proyecto capstone cargados, uno por sección.

POR QUÉ EXISTE. El enunciado (`ediciones/2026/proyecto-capstone.pdf`) promete un Colab con los seis
datasets listos para copiar. No es un notebook de clase: no enseña nada nuevo ni decide nada por el
equipo. Cada sección carga un archivo y muestra su forma, sus primeros registros y el reparto de su
target; explorar, limpiar y modelar es trabajo del equipo.

El estudiante copia a su propio Colab dos celdas: la de preparación (que define `BASE`) y la de su caso.

Se publica CON salidas. `make nb-build` solo acepta generadores `_build_NN_*`, así que se genera y
después se ejecuta en el sandbox, como el solucionario del repaso de la Solemne 2:

    conda run -n ust-ml python ediciones/2026/notebooks/_build_capstone_datasets.py
    conda run -n ust-ml python scripts/run_notebook_sandbox.py ediciones/2026/notebooks/capstone-datasets.ipynb \
        --keep-output ediciones/2026/notebooks/capstone-datasets.ipynb
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "capstone-datasets.ipynb"
LOGO = "data:image/svg+xml;base64," + base64.b64encode(
    (REPO_ROOT / "assets" / "logo-ust.svg").read_bytes()).decode("ascii")  # embebido: Colab no muestra el SVG externo
PAUTA = "https://sebastiancontz.github.io/ust-introduccion-machine-learning/ediciones/2026/proyecto-capstone.pdf"

# (variable, archivo, título, qué es un registro, target, cómo mirarlo)
CASOS = [
    ("telemarketing", "telemarketing_bancario.parquet", "Campañas de telemarketing bancario",
     "un cliente contactado en una campaña", "suscribio_deposito", "conteo"),
    ("compras", "compras_en_linea.parquet", "Intención de compra en una tienda en línea",
     "una sesión de navegación", "compro", "conteo"),
    ("trayectoria", "trayectoria_estudiantil.parquet", "Abandono y éxito académico en educación superior",
     "un estudiante", "situacion_final", "conteo"),
    ("bicicletas", "arriendo_bicicletas.parquet", "Arriendo de bicicletas compartidas",
     "una hora del sistema", "arriendos_total", "resumen"),
    ("colisiones", "colisiones_gb.parquet", "Colisiones viales 2024 (Gran Bretaña)",
     "una colisión con víctimas", "gravedad_colision", "conteo"),
    ("netbilling", "netbilling.parquet", "Netbilling: generación distribuida (Chile)",
     "una instalación inscrita", "potencia_kw", "resumen"),
]

ATRIBUCION = """
## Atribución de datos

Los seis archivos son obras derivadas: se tradujeron columnas y categorías al español y se resolvieron
tipos, sin limpiar ni quitar registros (colisiones conserva 15 de sus 44 columnas). El detalle de cada
modificación está en el anexo del [enunciado del proyecto]({pauta}).

- **Telemarketing bancario.** **Creador:** S. Moro, P. Rita y P. Cortez. **Fuente:** [UCI, *Bank Marketing*](https://archive.ics.uci.edu/dataset/222/bank+marketing). **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Modificación:** columnas y categorías en español; mes como número.
- **Compras en línea.** **Creador:** C. Sakar e Y. Kastro. **Fuente:** [UCI, *Online Shoppers Purchasing Intention Dataset*](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Modificación:** columnas y categorías en español; mes como número.
- **Trayectoria estudiantil.** **Creador:** V. Realinho, M. Vieira Martins, J. Machado y L. Baptista. **Fuente:** [UCI, *Predict Students' Dropout and Academic Success*](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Modificación:** columnas en español; categorías documentadas decodificadas.
- **Bicicletas compartidas.** **Creador:** H. Fanaee-T. **Fuente:** [UCI, *Bike Sharing*](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Modificación:** columnas y categorías en español; clima devuelto a sus unidades; sin la columna de índice.
- **Colisiones viales.** **Creador:** Department for Transport (Reino Unido). **Fuente:** [*Road safety open data*](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data). **Licencia:** [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). **Modificación:** 15 de 44 columnas, en español y decodificadas.
- **Netbilling.** **Creador:** Comisión Nacional de Energía. **Fuente:** [datos.gob.cl, *Netbilling*](https://datos.gob.cl/dataset/netbilling). **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Modificación:** potencia y fechas convertidas a número y fecha.
"""


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def seccion(n, variable, archivo, titulo, registro, target, mirar):
    reparto = (f'{variable}["{target}"].value_counts()  # cuántos registros hay de cada valor del target'
               if mirar == "conteo" else
               f'{variable}["{target}"].describe()  # resumen del target numérico')
    return [
        md(f"""
---

## Caso {n} · {titulo}

Un registro es **{registro}**; el target es `{target}`. Copien a su Colab la celda de preparación y la
celda siguiente.
"""),
        code(f"""
{variable} = pd.read_parquet(BASE + '{archivo}')  # carga el caso {n}
{variable}.shape
"""),
        code(f"{variable}.head()"),
        code(reparto),
    ]


def celdas() -> list[dict]:
    lista = [
        md(f"""
<div style="display:flex; align-items:center; gap:18px; text-align:left">
<img src="{LOGO}" width="100" alt="Logo de la Universidad Santo Tomás">
<div>
<p>Ingeniería en Información y Control de Gestión</p>
<p>Facultad de Economía y Negocios</p>
<p>Introducción a Machine Learning</p>
<p>Proyecto capstone: los seis casos</p>
</div>
</div>
"""),
        md(f"""
# Proyecto capstone · los seis casos cargados

Este notebook carga los seis datasets del proyecto. **No lo usen para trabajar**: copien a su propio
Colab la **celda de preparación** y la **celda de carga de su caso**, y desde ahí sigan ustedes.

Debajo de cada carga hay un primer vistazo —forma, primeros registros y reparto del target— solo para
reconocer el archivo. Explorar, preparar y modelar es parte del proyecto. Qué se pide y qué significa
cada columna están en el [enunciado del proyecto]({PAUTA}), que trae un diccionario de datos por caso.
"""),
        md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas; la celda queda para que el notebook corra también fuera
de Colab.
"""),
        code("""
%%capture
!pip install -q pandas pyarrow
"""),
        md("""
## Preparación: de dónde se leen los archivos

`BASE` es la carpeta pública con los datasets del curso. Esta celda va siempre primero en su Colab.
"""),
        code("""
import os
import pandas as pd

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO  # en Colab siempre se usa REPO
"""),
    ]
    for n, caso in enumerate(CASOS, 1):
        lista += seccion(n, *caso)
    lista.append(md(ATRIBUCION.format(pauta=PAUTA)))
    for i, celda in enumerate(lista):
        celda["id"] = f"c{i:02d}"  # nbformat 4.5+: ids deterministas
    return lista


def main() -> None:
    lista = celdas()
    nb = {"cells": lista,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    codigo = [c for c in lista if c["cell_type"] == "code"]
    fuente = "".join("".join(c["source"]) for c in codigo)
    assert all(archivo in fuente for _, archivo, *_ in CASOS)
    assert ".parquet" in fuente and "BASE + '" in fuente
    print(f"{SALIDA.relative_to(REPO_ROOT)}: {len(lista)} celdas "
          f"({len(lista) - len(codigo)} markdown / {len(codigo)} código)")


if __name__ == "__main__":
    main()
