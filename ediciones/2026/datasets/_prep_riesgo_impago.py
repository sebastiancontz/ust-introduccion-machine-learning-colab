"""Deriva la cartera de riesgo de impago de la clase 7 desde el XLS original de UCI 350.

POR QUE EXISTE, y por que no se reusa `morosidad_cartera.csv`. La clase 1 usa ese archivo para
RECONOCER encuadres sobre un mismo caso, y le basta con 600 registros. La clase 7 lo RESUELVE: mueve
un umbral y lee una matriz de confusion en cada posicion, asi que necesita celdas que no se muevan
por un caso. Con 600 registros el ROC-AUC oscila entre 0,58 y 0,78 segun la particion y la matriz de
prueba llega a tener una celda de 3 casos. Medido antes de disenar la clase; el detalle de las cinco
formulaciones evaluadas esta en `clases/_fuentes/semana-07-clasificacion.md`.

LOS DOS ARCHIVOS SON DISJUNTOS, y eso se comprueba. Este script reconstruye la muestra de 600 de la
clase 1 -misma llamada, misma semilla- y la EXCLUYE antes de muestrear. Asi se puede decir en clase
"misma fuente, otros clientes" sin que sea una promesa. La reconstruccion se verifica contra una
cifra publicada de ese archivo: 463 `no` y 137 `si`.

`morosidad_cartera.csv` NO SE TOCA: la clase 1 esta consolidada y sus cifras viven en su cuaderno.

QUE APORTA CADA COLUMNA NUEVA, y por que no bastan las 8 de la clase 1. Con solo `meses_mora`
mandando -un entero chico- las probabilidades se agrupan y el barrido del umbral queda escalonado:
entre 0,20 y 0,30 el recall no se mueve. La historia de tres meses, la mora acumulada y las dos
razones derivadas reparten la probabilidad y vuelven el barrido monotono, que es lo que la clase
necesita demostrar en vivo.

VARIABLES SENSIBLES EXCLUIDAS: `SEX` y `MARRIAGE` quedan fuera, igual que en la clase 1. La clase
discute que excluirlas no garantiza que ninguna otra columna actue como su sustituto.

UNIDAD DE OBSERVACION: un CLIENTE de tarjeta de credito observado en UN MES. Las predictoras
describen ese mes y los dos anteriores; el objetivo es el incumplimiento del mes SIGUIENTE.

DETERMINISTA: semilla fija, sin fechas del sistema. Correrlo dos veces produce el mismo archivo.

FALLA CERRADO: comprueba con `assert` cada cifra que el material afirma y, si alguna no se cumple,
NO escribe el archivo y termina con codigo de error.

Requiere el extra `datasets` del pyproject (xlrd) y el XLS original:
  MOROSIDAD_XLSX=/ruta/"default of credit card clients.xls" \
    conda run -n ust-ml python ediciones/2026/datasets/_prep_riesgo_impago.py
"""
import os
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
ORIGEN = Path(os.environ.get("MOROSIDAD_XLSX", "default of credit card clients.xls")).expanduser()
OUT = AQUI / "riesgo_impago.csv"

N_CLASE_9 = 6000
N_CLASE_1 = 600          # la muestra de `morosidad_cartera.csv`, que se excluye
SEMILLA_CLASE_1 = 42     # la que uso `_prep_morosidad_cartera.py`
SEMILLA_CLASE_9 = 9

EDUCACION = {1: "posgrado", 2: "universitaria", 3: "media", 0: "otra", 4: "otra", 5: "otra", 6: "otra"}
MESES_HISTORIA = ("PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")

if not ORIGEN.is_file():
    raise SystemExit(
        "Falta el XLS original de UCI 350. Descargalo de "
        "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients "
        "y define MOROSIDAD_XLSX con su ruta."
    )

crudo = pd.read_excel(ORIGEN, header=1)
assert crudo.shape == (30000, 25), f"el XLS de origen cambio de forma: {crudo.shape}"

# Las 600 de la clase 1: MISMA llamada y MISMA semilla que `_prep_morosidad_cartera.py`.
# `sample` elige por posicion, asi que no depende de que columnas tenga el DataFrame.
de_la_clase_1 = crudo.sample(n=N_CLASE_1, random_state=SEMILLA_CLASE_1).index
assert len(de_la_clase_1) == N_CLASE_1

# Se verifica la reconstruccion contra una cifra PUBLICADA de ese archivo, no contra si misma.
reparto = crudo.loc[de_la_clase_1, "default payment next month"].value_counts()
assert (int(reparto.get(0, 0)), int(reparto.get(1, 0))) == (463, 137), (
    f"la muestra reconstruida de la clase 1 no reproduce su reparto publicado: {reparto.to_dict()}"
)

disponible = crudo.drop(index=de_la_clase_1)
assert len(disponible) == 30000 - N_CLASE_1

m = disponible.sample(n=N_CLASE_9, random_state=SEMILLA_CLASE_9).sort_index()
assert len(m.index.intersection(de_la_clase_1)) == 0, "la muestra se solapa con la de la clase 1"

datos = pd.DataFrame({
    "id_cliente": [f"R-{i:04d}" for i in range(1, N_CLASE_9 + 1)],
    "edad": m["AGE"].to_numpy(),
    "nivel_educacion": m["EDUCATION"].map(EDUCACION).to_numpy(),
    "limite_credito": m["LIMIT_BAL"].to_numpy(),
    "meses_mora": m["PAY_0"].clip(lower=0).to_numpy(),
    "meses_mora_previo": m["PAY_2"].clip(lower=0).to_numpy(),
    "meses_mora_acumulados": m[list(MESES_HISTORIA)].clip(lower=0).sum(axis=1).to_numpy(),
    "monto_facturado_mes": m["BILL_AMT1"].to_numpy(),
    "monto_facturado_mes_anterior": m["BILL_AMT2"].to_numpy(),
    "monto_facturado_dos_meses_antes": m["BILL_AMT3"].to_numpy(),
    "monto_pagado_mes": m["PAY_AMT1"].to_numpy(),
    "monto_pagado_mes_anterior": m["PAY_AMT2"].to_numpy(),
    "monto_pagado_dos_meses_antes": m["PAY_AMT3"].to_numpy(),
})

# Dos razones derivadas. Reparten la probabilidad y por eso el barrido del umbral queda legible.
datos["uso_credito"] = (datos["monto_facturado_mes"] / datos["limite_credito"]).round(4)
datos["cobertura_pago"] = (
    datos["monto_pagado_mes"] / datos["monto_facturado_mes"].clip(lower=1)
).clip(0, 3).round(4)

datos["incumplio_pago"] = m["default payment next month"].map({0: "no", 1: "si"}).to_numpy()

# --- Comprobaciones de lo que el material de la clase afirma -------------------------------------
assert datos.shape == (N_CLASE_9, 16), f"forma inesperada: {datos.shape}"
assert datos.isna().sum().sum() == 0, "el archivo de la clase 7 no debe traer celdas vacias"
assert list(datos.columns)[0] == "id_cliente" and list(datos.columns)[-1] == "incumplio_pago"
assert datos["id_cliente"].is_unique

positivos = int((datos["incumplio_pago"] == "si").sum())
prevalencia = positivos / len(datos)
assert 0.20 <= prevalencia <= 0.24, f"la prevalencia se salio del rango esperado: {prevalencia:.3f}"

assert set(datos["nivel_educacion"]) <= set(EDUCACION.values())
assert datos["meses_mora"].min() == 0 and datos["meses_mora_previo"].min() == 0
assert datos["meses_mora_acumulados"].min() == 0
assert datos["cobertura_pago"].between(0, 3).all()
assert "SEX" not in datos.columns and "MARRIAGE" not in datos.columns
assert not any(c.lower() in {"sexo", "estado_civil"} for c in datos.columns)

datos.to_csv(OUT, index=False, encoding="utf-8")

print(f"{OUT.name}: {datos.shape[0]} registros x {datos.shape[1]} columnas")
print(f"incumplio_pago: {positivos} si / {len(datos) - positivos} no  ({prevalencia:.1%} de impago)")
print(f"piso de la clase mayoritaria: {1 - prevalencia:.3f} de exactitud, 0 de {positivos} detectados")
print(f"meses_mora: 0 a {int(datos['meses_mora'].max())} | "
      f"edad: {int(datos['edad'].min())} a {int(datos['edad'].max())}")
print(f"disjunto de morosidad_cartera.csv: si ({N_CLASE_1} registros excluidos antes de muestrear)")
