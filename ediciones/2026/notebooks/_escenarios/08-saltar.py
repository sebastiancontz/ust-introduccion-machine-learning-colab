"""SALTAR: el cuaderno publicado llega con `elegido = None`. Prueba no se abre y la evidencia del paso 3 sigue."""
# Vuelve a correr la celda REAL de la compuerta, leída del cuaderno copiado en el sandbox: se prueba el
# código publicado, con el estado vivo del kernel, y no una copia.
import glob
import json

_cuaderno = json.load(open(glob.glob("08-*.ipynb")[0], encoding="utf-8"))
CELDA_COMPUERTA = next("".join(c["source"]) for c in _cuaderno["cells"]
                       if c["cell_type"] == "code" and "pruebas_abiertas.append" in "".join(c["source"]))


def correr_compuerta(eleccion):
    globals()["elegido"] = eleccion
    exec(CELDA_COMPUERTA, globals())

assert elegido is None, f"el cuaderno publicado ya trae una elección: {elegido!r}"
assert pruebas_abiertas == [], f"prueba se abrió sin elegir: {pruebas_abiertas}"
assert "auc_prueba" not in globals(), "existe una cifra de prueba sin que nadie eligiera"
assert comparacion.shape == (5, 5), "la tabla de comparación del paso 3 no está completa"
correr_compuerta("bosqe")  # una elección mal escrita tampoco abre prueba
assert pruebas_abiertas == [] and "auc_prueba" not in globals()
print("saltar: OK")
