"""Prepara la version de clase 1 del dataset de morosidad (UCI 350) -> CSV en espanol."""
import os

import pandas as pd

SP = "/tmp/claude-1000/-home-sebastian-Documents-ust-introduccion-machine-learning/c9c00a93-f86d-4a1c-a5bc-5271ce30ed28/scratchpad"
OUT = ("/home/sebastian/Documents/ust-introduccion-machine-learning/"
       "introduccion-machine-learning/2026/datasets/morosidad_cartera.csv")

raw = pd.read_excel(f"{SP}/default of credit card clients.xls", header=1)

EDU = {1: "posgrado", 2: "universitaria", 3: "media", 4: "otra", 0: "otra", 5: "otra", 6: "otra"}

df = pd.DataFrame({
    "id_cliente": raw["ID"],
    "edad": raw["AGE"],
    "nivel_educacion": raw["EDUCATION"].map(EDU),
    "limite_credito": raw["LIMIT_BAL"],
    "monto_facturado_mes": raw["BILL_AMT1"],
    "monto_pagado_mes": raw["PAY_AMT1"],
    "meses_mora": raw["PAY_0"].clip(lower=0),
    "incumplio_pago": raw["default payment next month"].map({0: "no", 1: "si"}),
})

muestra = df.sample(n=600, random_state=42).sort_values("id_cliente").reset_index(drop=True)
muestra["id_cliente"] = [f"C-{i:04d}" for i in range(1, len(muestra) + 1)]
muestra.to_csv(OUT, index=False, encoding="utf-8")

print(muestra.shape, "nulos:", muestra.isna().sum().sum())
print(muestra.head(5).to_string(index=False))
print("\nincumplio_pago:", muestra["incumplio_pago"].value_counts().to_dict())
print("nivel_educacion:", muestra["nivel_educacion"].value_counts().to_dict())
print("meses_mora:", sorted(muestra["meses_mora"].unique()))
print("\ndtypes:")
print(muestra.dtypes.to_string())
print("\nbytes:", os.path.getsize(OUT))
