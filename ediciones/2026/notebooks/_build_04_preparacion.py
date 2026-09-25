#!/usr/bin/env python3
"""Genera los DOS notebooks de la sesión 4: parte A y parte B.

La sesión 4 fusiona las clases 4 y 5 en un bloque presencial de 105 minutos, pero su cuaderno vuelve a
ser DOS por decisión del docente (2026-09-07): un solo notebook de 134 celdas es difícil de navegar en
clase, y las dos mitades se trabajan en momentos distintos, separadas por el receso.

  · `_celdas_sesion04_corregir.py`     -> `04-preparacion.ipynb`         parte A, decidir sobre lo que está MAL
  · `_celdas_sesion04_representar.py`  -> `05-ingenieria-variables.ipynb` parte B, transformar lo que ya está BIEN

Cada módulo es autosuficiente y escribe su propio archivo: trae su cabecera institucional, su título,
su bloque de entorno y su instalación —A con pandas y pyarrow, B agregando numpy y scikit-learn—. Este
script no los empalma: solo los ejecuta a los dos y comprueba, con asserts, que sigan siendo
autosuficientes. Si alguna mitad pierde su instalación o su cabecera, esto falla en vez de publicar un
cuaderno que no corre en Colab.

El contrato de `curso_checks` lo declara solo la parte B, que es la que tiene marcadores de acción; la
parte A no los tiene y tampoco los tenía antes de la fusión.

POR QUÉ LA PARTE B CONSERVA EL NOMBRE `05-ingenieria-variables.ipynb`: su URL de Colab ya está
publicada y enlazada desde el sitio. Renumerarla rompería enlaces que los estudiantes pueden tener
abiertos, y no gana nada.

Uso:  python _build_04_preparacion.py
      y después, para refrescar salidas, uno por uno:
      python scripts/run_notebook_sandbox.py <notebook> --keep-output <tmp> && mv <tmp> <notebook>
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]


def _modulo(nombre: str):
    ruta = AQUI / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def _autosuficiente(mod, nombre: str, install: str) -> None:
    """Comprueba que la mitad pueda publicarse sola: cabecera, título e instalación propias."""
    celdas = mod.CELDAS
    assert celdas[0]["cell_type"] == "markdown" and "display:flex" in "".join(celdas[0]["source"]), \
        f"{nombre}: la celda 0 ya no es la cabecera institucional"
    assert "".join(celdas[1]["source"]).lstrip().startswith("# "), \
        f"{nombre}: la celda 1 ya no es su título"
    instal = [c for c in celdas
              if c["cell_type"] == "code" and "".join(c["source"]).strip() == install]
    assert len(instal) == 1, \
        f"{nombre}: esperaba exactamente 1 celda de instalación con {install!r}, hay {len(instal)}"


def main() -> None:
    corregir = _modulo("_celdas_sesion04_corregir")
    representar = _modulo("_celdas_sesion04_representar")

    _autosuficiente(corregir, "parte A", "%%capture\n!pip install -q pandas pyarrow")
    _autosuficiente(representar, "parte B",
                    "%%capture\n!pip install -q pandas numpy pyarrow scikit-learn")

    corregir.main()
    representar.main()


if __name__ == "__main__":
    main()
