"""COMPLETAR: el grupo elige un modelo y abre prueba una vez; puede volver a correr la celda con el mismo."""
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

correr_compuerta("logistica")
assert pruebas_abiertas == ["logistica"], pruebas_abiertas
assert 0.6 < auc_prueba < 0.9, auc_prueba
primera = auc_prueba
correr_compuerta("logistica")  # repetir con el mismo modelo no agrega aperturas ni cambia la cifra
assert pruebas_abiertas == ["logistica"] and auc_prueba == primera
assert comparacion.shape == (5, 5), "la evidencia del paso 3 desapareció"
print("completar: OK")
