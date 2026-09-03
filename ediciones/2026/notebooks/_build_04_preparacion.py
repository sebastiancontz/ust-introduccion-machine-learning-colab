#!/usr/bin/env python3
"""Genera el notebook ÚNICO de la sesión 4, que va de corregir a representar.

La sesión 4 fusiona las clases 4 y 5 en un solo bloque presencial de 105 minutos, así que su
notebook también es uno solo. Sus celdas viven en dos módulos, uno por mitad:

  · `_celdas_sesion04_corregir.py`     — decidir sobre lo que está MAL (era la clase 4)
  · `_celdas_sesion04_representar.py`  — transformar lo que ya está BIEN (era la clase 5)

Se mantienen separados a propósito: son las dos mitades pedagógicas de la sesión y cada una se
edita sin tocar la otra. Las versiones de esas mitades como notebooks SUELTOS, para una edición que
vuelva a dictarlas en dos sesiones, están archivadas en `_sesiones-separadas/notebooks/`.

El empalme quita del segundo módulo lo que ya trae el primero —la cabecera institucional y el bloque
de preparación del entorno— y convierte su título en el divisor de la segunda mitad. La instalación
del primero se amplía a la unión de las dos, porque la segunda mitad necesita numpy y scikit-learn.

Uso:  python _build_04_preparacion.py   y después, para refrescar salidas:
      python scripts/run_notebook_sandbox.py <notebook> --keep-output <tmp> && mv <tmp> <notebook>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "04-preparacion.ipynb"


def _modulo(nombre: str):
    ruta = AQUI / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


CORREGIR = _modulo("_celdas_sesion04_corregir")
REPRESENTAR = _modulo("_celdas_sesion04_representar")

# ── empalme, con asserts: si alguna mitad cambia su cabecera, esto falla en vez de duplicarla ──
INSTALL_VIEJO = "%%capture\n!pip install -q pandas pyarrow"
INSTALL_NUEVO = "%%capture\n!pip install -q pandas numpy pyarrow scikit-learn"

DIVISOR_SEGUNDA_MITAD = """---

# Segunda parte · representar lo que ya está bien

Hasta acá **corregimos** lo que estaba mal en el archivo y lo dejamos escrito en la bitácora. Lo que
sigue es la otra pregunta de la sesión: *¿esto está correcto pero en un formato que el modelo no
aprovecha?* El archivo es el mismo, ya con las decisiones aplicadas.
"""


def _celdas() -> list[dict]:
    a = [dict(c) for c in CORREGIR.CELDAS]
    b = [dict(c) for c in REPRESENTAR.CELDAS]

    # 1) la instalación de la primera mitad pasa a cubrir las dos
    instal = [i for i, c in enumerate(a)
              if c["cell_type"] == "code" and "".join(c["source"]).strip() == INSTALL_VIEJO]
    assert len(instal) == 1, f"esperaba 1 celda de instalación en la primera mitad, hay {len(instal)}"
    a[instal[0]] = {**a[instal[0]], "source": INSTALL_NUEVO}

    # 2) del segundo módulo se descartan cabecera institucional, título, encabezado de entorno e
    #    instalación: los cuatro ya están en la primera mitad
    assert b[0]["cell_type"] == "markdown" and "display:flex" in "".join(b[0]["source"]), \
        "la celda 0 de la segunda mitad ya no es la cabecera institucional"
    assert "".join(b[1]["source"]).lstrip().startswith("# 05 ·"), \
        "la celda 1 de la segunda mitad ya no es su título"
    assert "".join(b[2]["source"]).strip() == "## Preparación del entorno", \
        "la celda 2 de la segunda mitad ya no es el encabezado del entorno"
    assert b[3]["cell_type"] == "code" and "pip install" in "".join(b[3]["source"]), \
        "la celda 3 de la segunda mitad ya no es su instalación"

    divisor = {"cell_type": "markdown", "metadata": {}, "source": DIVISOR_SEGUNDA_MITAD}
    return a + [divisor] + b[4:]


def _contrato(celdas: list[dict]) -> dict:
    """Calcula el contrato con el MISMO código que lo verifica.

    El contrato de la segunda mitad contaba las llamadas con `str.count("print(")`, que sobre una sola
    mitad coincidía porque todos sus `print` llevaban un argumento. Al fusionar deja de coincidir: la
    primera mitad tiene `print` de varios argumentos y el checker cuenta por AST, no por texto. En vez
    de replicar esa lógica, se importa la del checker: así el contrato no puede divergir de su gate.
    """
    import ast
    import importlib.util as _il

    ruta = REPO_ROOT / "scripts" / "check_notebook.py"
    spec = _il.spec_from_file_location("_check_notebook", ruta)
    chk = _il.module_from_spec(spec)
    sys.modules["_check_notebook"] = chk
    spec.loader.exec_module(chk)

    llamadas = []
    for cell in celdas:
        if cell.get("cell_type") != "code" or chk._sandbox_scenario(cell):
            continue
        tree, _ = chk._parse_code(cell)
        if tree is not None:
            llamadas.extend(chk._calls(tree))

    def cuantas(funcion: str, argumentos: int) -> int:
        return sum(chk._call_name(c.func) == funcion
                   and len(c.args) + len(c.keywords) == argumentos
                   for c in llamadas)

    base = REPRESENTAR.contrato(celdas)
    base["forma"]["llamadas"] = [
        {"funcion": "error_libras", "argumentos": 2, "cantidad": cuantas("error_libras", 2)},
        {"funcion": "print", "argumentos": 1, "cantidad": cuantas("print", 1)},
    ]
    return base


def main() -> None:
    celdas = REPRESENTAR.con_ids(_celdas())
    nb = {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "curso_contrato": _contrato(celdas),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    forma = nb["metadata"]["curso_contrato"]["forma"]
    print(f"escrito {SALIDA.relative_to(REPO_ROOT)}")
    print(f"  {len(celdas)} celdas · {forma['markdown']} markdown · {forma['codigo']} código")
    print(f"  contrato: mediana {nb['metadata']['curso_contrato']['max_mediana_lineas_codigo'] - 1}"
          f" · llamadas {forma['llamadas']}")


if __name__ == "__main__":
    main()
