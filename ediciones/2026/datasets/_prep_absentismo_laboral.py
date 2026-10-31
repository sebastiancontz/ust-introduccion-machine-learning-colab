"""Prepara el dataset de absentismo de la clase 2 (UCI 445) -> CSV en espanol.

Autocontenido: descarga el original desde UCI, reconstruye el eje temporal y escribe el CSV.
Ejecutar desde cualquier directorio:  python _prep_absentismo_laboral.py

El aporte no trivial es la RECONSTRUCCION DEL PERIODO. El original no trae anio, solo
`Month of absence` de 1 a 12, asi que a primera vista no hay linea de tiempo. Si la hay:
`Work load Average/day` es una cifra mensual de planta con 38 valores unicos que forman 38
pares unicos con el mes, aparecen en bloques contiguos sin solaparse y, en el orden del
archivo, dibujan una secuencia mensual corrida de julio a julio. El archivo esta ordenado
cronologicamente, asi que el bloque n-esimo de carga es el mes n-esimo desde 2007-07.
El script VERIFICA esas tres condiciones antes de confiar en ellas y aborta si no se cumplen.
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/445/absenteeism+at+work.zip"
INICIO = pd.Period("2007-07", freq="M")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "absentismo_laboral.csv")

# ICD-10: los codigos 1-21 son capitulos de la clasificacion; 22-28, motivos administrativos.
MOTIVO = {
    0: "sin motivo registrado",
    1: "enfermedades infecciosas y parasitarias",
    2: "tumores",
    3: "enfermedades de la sangre",
    4: "enfermedades endocrinas y metabolicas",
    5: "trastornos mentales y del comportamiento",
    6: "enfermedades del sistema nervioso",
    7: "enfermedades de los ojos",
    8: "enfermedades del oido",
    9: "enfermedades del sistema circulatorio",
    10: "enfermedades del sistema respiratorio",
    11: "enfermedades del sistema digestivo",
    12: "enfermedades de la piel",
    13: "enfermedades del sistema musculoesqueletico",
    14: "enfermedades del sistema genitourinario",
    15: "embarazo, parto y puerperio",
    16: "afecciones del periodo perinatal",
    17: "malformaciones congenitas",
    18: "sintomas no clasificados en otra parte",
    19: "lesiones y envenenamientos",
    20: "causas externas de morbilidad",
    21: "factores que influyen en el estado de salud",
    22: "control medico posterior",
    23: "consulta medica",
    24: "donacion de sangre",
    25: "examen de laboratorio",
    26: "ausencia injustificada",
    27: "kinesiologia",
    28: "consulta dental",
}
DIA = {2: "lunes", 3: "martes", 4: "miercoles", 5: "jueves", 6: "viernes"}
# El registro es de una empresa brasilena: las estaciones siguen el hemisferio sur.
ESTACION = {1: "verano", 2: "otonio", 3: "invierno", 4: "primavera"}
EDUCACION = {1: "media", 2: "universitaria", 3: "posgrado", 4: "magister o doctorado"}
SI_NO = {0: "no", 1: "si"}

RENOMBRE = {
    "ID": "id_empleado",
    "Reason for absence": "motivo_ausencia",
    "Month of absence": "mes",
    "Day of the week": "dia_semana",
    "Seasons": "estacion",
    "Transportation expense": "gasto_transporte",
    "Distance from Residence to Work": "distancia_casa_trabajo_km",
    "Service time": "antiguedad_anios",
    "Age": "edad",
    "Work load Average/day": "carga_trabajo_promedio_dia",
    "Hit target": "cumplimiento_meta_pct",
    "Disciplinary failure": "falta_disciplinaria",
    "Education": "nivel_educacion",
    "Son": "hijos",
    "Social drinker": "bebedor_social",
    "Social smoker": "fumador_social",
    "Pet": "mascotas",
    "Weight": "peso_kg",
    "Height": "altura_cm",
    "Body mass index": "indice_masa_corporal",
    "Absenteeism time in hours": "horas_ausencia",
}


def descargar():
    with urllib.request.urlopen(URL) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    with z.open("Absenteeism_at_work.csv") as f:
        raw = pd.read_csv(f, sep=";")
    raw.columns = [c.strip() for c in raw.columns]
    return raw


def reconstruir_periodo(raw):
    """Devuelve una Serie de periodos (str 'YYYY-MM'), vacia donde el mes es 0."""
    fechado = raw[raw["mes"] > 0]
    carga = "carga_trabajo_promedio_dia"

    # (1) cada valor de carga corresponde a un unico mes calendario
    meses = fechado.groupby(carga)["mes"].nunique()
    assert (meses == 1).all(), "una carga cubre mas de un mes: el supuesto no se sostiene"

    bloques = fechado.groupby(carga).agg(prim=("orden", "min"), ult=("orden", "max"))
    bloques = bloques.sort_values("prim")

    # (2) los bloques no se solapan en el orden del archivo
    solapes = (bloques["prim"].shift(-1) < bloques["ult"]).sum()
    assert solapes == 0, f"{solapes} bloques de carga se solapan: el archivo no esta ordenado"

    # (3) la secuencia de meses que implican es corrida, sin saltos
    seq = [fechado.loc[fechado[carga] == c, "mes"].iloc[0] for c in bloques.index]
    saltos = [i for i in range(len(seq) - 1) if (seq[i] % 12) + 1 != seq[i + 1]]
    assert not saltos, f"la secuencia mensual salta en las posiciones {saltos}"

    k = {c: i for i, c in enumerate(bloques.index)}
    return raw[carga].map(lambda c: str(INICIO + k[c]) if c in k else "")


raw = descargar().reset_index(names="orden").rename(columns=RENOMBRE)

df = pd.DataFrame({
    "id_empleado": raw["id_empleado"].map(lambda i: f"E-{i:02d}"),
    "periodo": reconstruir_periodo(raw),
    "mes": raw["mes"],
    "dia_semana": raw["dia_semana"].map(DIA),
    "estacion": raw["estacion"].map(ESTACION),
    "motivo_ausencia": raw["motivo_ausencia"].map(MOTIVO),
    "horas_ausencia": raw["horas_ausencia"],
    "edad": raw["edad"],
    "antiguedad_anios": raw["antiguedad_anios"],
    "nivel_educacion": raw["nivel_educacion"].map(EDUCACION),
    "hijos": raw["hijos"],
    "mascotas": raw["mascotas"],
    "distancia_casa_trabajo_km": raw["distancia_casa_trabajo_km"],
    "gasto_transporte": raw["gasto_transporte"],
    "carga_trabajo_promedio_dia": raw["carga_trabajo_promedio_dia"],
    "cumplimiento_meta_pct": raw["cumplimiento_meta_pct"],
    "falta_disciplinaria": raw["falta_disciplinaria"].map(SI_NO),
    "bebedor_social": raw["bebedor_social"].map(SI_NO),
    "fumador_social": raw["fumador_social"].map(SI_NO),
    "peso_kg": raw["peso_kg"],
    "altura_cm": raw["altura_cm"],
    "indice_masa_corporal": raw["indice_masa_corporal"],
})

df.to_csv(OUT, index=False, encoding="utf-8")

fechado = df[df["periodo"] != ""]
print(df.shape, "| nulos:", df.isna().sum().sum(), "| duplicados exactos:", df.duplicated().sum())
print("empleados:", df["id_empleado"].nunique(), "| con ausencia fechada:", fechado["id_empleado"].nunique())
print("periodos:", fechado["periodo"].nunique(), "de", fechado["periodo"].min(), "a", fechado["periodo"].max())
print("filas sin periodo (mes 0):", (df["periodo"] == "").sum())
print("motivos distintos:", df["motivo_ausencia"].nunique())
print("horas_ausencia: min %d max %d media %.2f" % (
    df["horas_ausencia"].min(), df["horas_ausencia"].max(), df["horas_ausencia"].mean()))
print("\nbytes:", os.path.getsize(OUT))
print(df.head(3).to_string(index=False))
