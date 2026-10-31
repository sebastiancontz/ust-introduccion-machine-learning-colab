"""VOLVER: después de abrir prueba para un modelo, el grupo cambia de idea. La compuerta no abre prueba para
el segundo y conserva la decisión y la cifra del primero."""
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

correr_compuerta("bosque")
assert pruebas_abiertas == ["bosque"], pruebas_abiertas
cifra_bosque = auc_prueba
correr_compuerta("arbol")
assert pruebas_abiertas == ["bosque"], f"prueba se abrió para un segundo modelo: {pruebas_abiertas}"
assert auc_prueba == cifra_bosque, "la cifra de prueba cambió al cambiar de modelo"
assert comparacion.shape == (5, 5), "la evidencia del paso 4 desapareció"
print("volver: OK")
