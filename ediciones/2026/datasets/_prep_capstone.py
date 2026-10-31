"""Prepara los seis casos del proyecto capstone (enunciado: `ediciones/2026/proyecto-capstone.pdf`).

POR QUE EXISTE. La pauta promete cada caso «preparado para el curso»: columnas en español y tipos
resueltos, como el resto de los datasets. Esa preparación es MINIMA a proposito: traduce, decodifica lo
que la fuente documenta y convierte tipos. NO limpia, no deduplica, no imputa y no quita variables:
decidir eso es el trabajo de cada equipo (bloques 3 y 4 del canvas de la sesión 11). La única excepción
es colisiones, que conserva el recorte de 15 columnas del diplomado.

SEIS ARCHIVOS, uno por caso, en Parquet (los tipos viajan con el archivo):

- `telemarketing_bancario.parquet` · UCI 222, `bank-full.csv` · 45.211 × 17
- `compras_en_linea.parquet` · UCI 468 · 12.330 × 18
- `trayectoria_estudiantil.parquet` · UCI 697 · 4.424 × 37
- `arriendo_bicicletas.parquet` · UCI 275, `hour.csv` · 17.379 × 16 (sin `instant`, que es el índice)
- `colisiones_gb.parquet` · DfT, colisiones 2024 · 100.927 × 15 (recorte y decodificación del diplomado,
  con el distrito ONS en vez del código antiguo, vacío en 2024)
- `netbilling.parquet` · CNE vía datos.gob.cl · 39.636 × 9

Las fuentes de colisiones y Netbilling se actualizan: los `assert` de forma fijan la versión con que se
preparó (Netbilling con corte `fecha_act` 2025-12-01). Si una fuente cambia, el script falla cerrado.

DETERMINISTA: sin muestreo ni fechas del sistema. Descarga los originales a `CAPSTONE_CRUDO`
(por defecto `~/.cache/ust-ml-capstone`) solo si faltan. Uso:

    conda run -n ust-ml python ediciones/2026/datasets/_prep_capstone.py
"""
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
CRUDO = Path(os.environ.get("CAPSTONE_CRUDO", "~/.cache/ust-ml-capstone")).expanduser()
FUENTES = {
    "uci_222.zip": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "uci_468.zip": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
    "uci_697.zip": "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip",
    "uci_275.zip": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "colisiones_2024.csv": "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2024.csv",
    "netbilling.csv": "https://datos.gob.cl/dataset/f68a4c2b-605d-44fb-a485-c8a9a09b0e4e/resource/564d9030-c410-4633-b4d7-e67df8ca9d37/download/netbilling.csv",
}
SI_NO = {"yes": "si", "no": "no"}
BINARIO = {1: "si", 0: "no"}


def crudo(nombre):
    ruta = CRUDO / nombre
    if not ruta.is_file():
        CRUDO.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FUENTES[nombre], ruta)
    return ruta


def decodificar(df, mapas):
    for col, mapa in mapas.items():
        sin_mapa = set(df[col].dropna().unique()) - set(mapa)
        assert not sin_mapa, f"{col}: valores sin traducción {sin_mapa}"
        df[col] = df[col].map(mapa)
    return df


def guardar(df, nombre, forma):
    assert df.shape == forma, f"{nombre}: forma {df.shape}, se esperaba {forma}"
    assert all(c == c.lower() and " " not in c for c in df.columns), f"{nombre}: columnas fuera de snake_case"
    df.to_parquet(AQUI / nombre, engine="pyarrow", compression="zstd", index=False)
    print(f"{nombre:34s} {df.shape[0]:>7} × {df.shape[1]:>2}  {(AQUI / nombre).stat().st_size / 1e6:.2f} MB")


# ── 1 · telemarketing bancario (UCI 222) ─────────────────────────────────────────────────────────
def telemarketing():
    externo = zipfile.ZipFile(crudo("uci_222.zip"))
    df = pd.read_csv(zipfile.ZipFile(io.BytesIO(externo.read("bank.zip"))).open("bank-full.csv"), sep=";")
    df = df.rename(columns={
        "age": "edad", "job": "ocupacion", "marital": "estado_civil", "education": "educacion",
        "default": "credito_en_mora", "balance": "saldo_promedio_anual", "housing": "credito_hipotecario",
        "loan": "credito_consumo", "contact": "medio_contacto", "day": "dia_ultimo_contacto",
        "month": "mes_ultimo_contacto", "duration": "duracion_ultimo_contacto_seg",
        "campaign": "contactos_esta_campania", "pdays": "dias_desde_campania_anterior",
        "previous": "contactos_previos", "poutcome": "resultado_campania_anterior", "y": "suscribio_deposito",
    })
    decodificar(df, {
        "ocupacion": {"management": "gerencia", "technician": "tecnico", "entrepreneur": "emprendedor",
                      "blue-collar": "obrero", "unknown": "desconocido", "retired": "jubilado",
                      "admin.": "administrativo", "services": "servicios", "self-employed": "independiente",
                      "unemployed": "desempleado", "housemaid": "trabajo_domestico", "student": "estudiante"},
        "estado_civil": {"married": "casado", "single": "soltero", "divorced": "divorciado"},
        "educacion": {"primary": "basica", "secondary": "media", "tertiary": "superior", "unknown": "desconocido"},
        "credito_en_mora": SI_NO, "credito_hipotecario": SI_NO, "credito_consumo": SI_NO,
        "medio_contacto": {"cellular": "celular", "telephone": "telefono_fijo", "unknown": "desconocido"},
        "mes_ultimo_contacto": {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
                                "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12},
        "resultado_campania_anterior": {"success": "exito", "failure": "fracaso", "other": "otro",
                                        "unknown": "desconocido"},
        "suscribio_deposito": SI_NO,
    })
    assert round((df["suscribio_deposito"] == "si").mean(), 3) == 0.117
    guardar(df, "telemarketing_bancario.parquet", (45211, 17))


# ── 2 · intención de compra en línea (UCI 468) ───────────────────────────────────────────────────
def compras():
    df = pd.read_csv(zipfile.ZipFile(crudo("uci_468.zip")).open("online_shoppers_intention.csv"))
    df = df.rename(columns={
        "Administrative": "paginas_administrativas", "Administrative_Duration": "duracion_administrativas_seg",
        "Informational": "paginas_informativas", "Informational_Duration": "duracion_informativas_seg",
        "ProductRelated": "paginas_producto", "ProductRelated_Duration": "duracion_producto_seg",
        "BounceRates": "tasa_rebote", "ExitRates": "tasa_salida", "PageValues": "valor_paginas",
        "SpecialDay": "cercania_fecha_especial", "Month": "mes", "OperatingSystems": "sistema_operativo",
        "Browser": "navegador", "Region": "region", "TrafficType": "tipo_trafico",
        "VisitorType": "tipo_visitante", "Weekend": "fin_de_semana", "Revenue": "compro",
    })
    decodificar(df, {
        "mes": {"Feb": 2, "Mar": 3, "May": 5, "June": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
                "Nov": 11, "Dec": 12},
        "tipo_visitante": {"Returning_Visitor": "recurrente", "New_Visitor": "nuevo", "Other": "otro"},
        "fin_de_semana": {True: "si", False: "no"}, "compro": {True: "si", False: "no"},
    })
    guardar(df, "compras_en_linea.parquet", (12330, 18))


# ── 3 · trayectoria estudiantil (UCI 697) ────────────────────────────────────────────────────────
CARRERAS = {
    33: "tecnologias_biocombustibles", 171: "diseno_animacion_multimedia", 8014: "servicio_social_vespertino",
    9003: "agronomia", 9070: "diseno_comunicacion", 9085: "enfermeria_veterinaria",
    9119: "ingenieria_informatica", 9130: "equinocultura", 9147: "administracion",
    9238: "servicio_social", 9254: "turismo", 9500: "enfermeria", 9556: "higiene_oral",
    9670: "publicidad_marketing", 9773: "periodismo_comunicacion", 9853: "educacion_basica",
    9991: "administracion_vespertino",
}
NACIONALIDADES = {
    1: "portuguesa", 2: "alemana", 6: "espanola", 11: "italiana", 13: "neerlandesa", 14: "inglesa",
    17: "lituana", 21: "angolena", 22: "caboverdiana", 24: "guineana", 25: "mozambiquena",
    26: "santotomense", 32: "turca", 41: "brasilena", 62: "rumana", 100: "moldava", 101: "mexicana",
    103: "ucraniana", 105: "rusa", 108: "cubana", 109: "colombiana",
}


def trayectoria():
    df = pd.read_csv(zipfile.ZipFile(crudo("uci_697.zip")).open("data.csv"), sep=";")
    df.columns = [c.strip() for c in df.columns]
    semestre = {"1st": "sem1", "2nd": "sem2"}
    cuenta = {"credited": "convalidadas", "enrolled": "inscritas", "evaluations": "evaluaciones",
              "approved": "aprobadas", "grade": "nota_promedio", "without evaluations": "sin_evaluaciones"}
    renombre = {
        "Marital status": "estado_civil", "Application mode": "modalidad_postulacion",
        "Application order": "orden_preferencia", "Course": "carrera",
        "Daytime/evening attendance": "jornada", "Previous qualification": "estudios_previos",
        "Previous qualification (grade)": "nota_estudios_previos", "Nacionality": "nacionalidad",
        "Mother's qualification": "estudios_madre", "Father's qualification": "estudios_padre",
        "Mother's occupation": "ocupacion_madre", "Father's occupation": "ocupacion_padre",
        "Admission grade": "puntaje_admision", "Displaced": "vive_fuera_de_su_ciudad",
        "Educational special needs": "necesidades_educativas_especiales", "Debtor": "deudor",
        "Tuition fees up to date": "arancel_al_dia", "Gender": "sexo", "Scholarship holder": "becado",
        "Age at enrollment": "edad_al_matricularse", "International": "internacional",
        "Unemployment rate": "tasa_desempleo", "Inflation rate": "tasa_inflacion", "GDP": "pib",
        "Target": "situacion_final",
    }
    for s, sufijo in semestre.items():
        for c, nombre in cuenta.items():
            renombre[f"Curricular units {s} sem ({c})"] = f"asignaturas_{sufijo}_{nombre}"
    df = df.rename(columns=renombre)
    decodificar(df, {
        "estado_civil": {1: "soltero", 2: "casado", 3: "viudo", 4: "divorciado", 5: "union_de_hecho",
                         6: "separado"},
        "carrera": CARRERAS, "jornada": {1: "diurna", 0: "vespertina"}, "nacionalidad": NACIONALIDADES,
        "sexo": {1: "masculino", 0: "femenino"},
        "vive_fuera_de_su_ciudad": BINARIO, "necesidades_educativas_especiales": BINARIO, "deudor": BINARIO,
        "arancel_al_dia": BINARIO, "becado": BINARIO, "internacional": BINARIO,
        "situacion_final": {"Dropout": "abandono", "Enrolled": "matriculado", "Graduate": "titulado"},
    })
    assert df["situacion_final"].value_counts().to_dict() == {"titulado": 2209, "abandono": 1421, "matriculado": 794}
    guardar(df, "trayectoria_estudiantil.parquet", (4424, 37))


# ── 4 · arriendo de bicicletas (UCI 275, por hora) ───────────────────────────────────────────────
def bicicletas():
    df = pd.read_csv(zipfile.ZipFile(crudo("uci_275.zip")).open("hour.csv")).drop(columns="instant")
    # la fuente normaliza estas cuatro dividiendo por un máximo; se devuelven a sus unidades (Readme de UCI)
    df["temp"] = (df["temp"] * 41).round(2)
    df["atemp"] = (df["atemp"] * 50).round(2)
    df["hum"] = (df["hum"] * 100).round(0).astype(int)
    df["windspeed"] = (df["windspeed"] * 67).round(2)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df["yr"] = df["yr"] + 2011
    df = df.rename(columns={
        "dteday": "fecha", "season": "estacion", "yr": "anio", "mnth": "mes", "hr": "hora",
        "holiday": "feriado", "weekday": "dia_semana", "workingday": "dia_laboral", "weathersit": "clima",
        "temp": "temperatura_c", "atemp": "sensacion_termica_c", "hum": "humedad_pct",
        "windspeed": "velocidad_viento", "casual": "arriendos_ocasionales",
        "registered": "arriendos_registrados", "cnt": "arriendos_total",
    })
    decodificar(df, {
        # estaciones del hemisferio norte, codificadas por la fuente: 1 primavera … 4 invierno
        "estacion": {1: "primavera", 2: "verano", 3: "otono", 4: "invierno"},
        "feriado": BINARIO, "dia_laboral": BINARIO,
        "dia_semana": {0: "domingo", 1: "lunes", 2: "martes", 3: "miercoles", 4: "jueves", 5: "viernes",
                       6: "sabado"},
        "clima": {1: "despejado", 2: "neblina_nublado", 3: "lluvia_o_nieve_ligera", 4: "lluvia_o_nieve_intensa"},
    })
    assert (df["arriendos_ocasionales"] + df["arriendos_registrados"] == df["arriendos_total"]).all()
    guardar(df, "arriendo_bicicletas.parquet", (17379, 16))


# ── 5 · colisiones viales GB 2024 (DfT) · recorte y decodificación del diplomado ────────────────
COLISIONES = {
    "collision_index": "id_colision", "collision_year": "anio", "date": "fecha", "day_of_week": "dia_semana",
    "time": "hora", "collision_severity": "gravedad_colision", "number_of_vehicles": "numero_vehiculos",
    "number_of_casualties": "numero_victimas", "road_type": "tipo_via", "speed_limit": "limite_velocidad",
    "light_conditions": "condiciones_luz", "weather_conditions": "condiciones_clima",
    "road_surface_conditions": "condiciones_superficie", "urban_or_rural_area": "area_urbana_rural",
    # en 2024 `local_authority_district` vale -1 en todos los registros; se usa el código ONS del distrito
    "local_authority_ons_district": "distrito_ons",
}
COLISIONES_DECODIFICAR = {
    "gravedad_colision": {1: "fatal", 2: "grave", 3: "leve"},
    "area_urbana_rural": {1: "urbana", 2: "rural", 3: "sin_asignar"},
    "dia_semana": {1: "domingo", 2: "lunes", 3: "martes", 4: "miercoles", 5: "jueves", 6: "viernes", 7: "sabado"},
    "tipo_via": {1: "rotonda", 2: "un_sentido", 3: "doble_calzada", 6: "calzada_simple", 7: "rampa",
                 9: "desconocido", 12: "un_sentido_rampa", -1: None},
    "condiciones_luz": {1: "dia", 4: "noche_con_luz", 5: "noche_sin_luz", 6: "noche_sin_alumbrado",
                        7: "noche_luz_desconocida", -1: None},
    "condiciones_clima": {1: "despejado", 2: "lluvia", 3: "nieve", 4: "despejado_viento", 5: "lluvia_viento",
                          6: "nieve_viento", 7: "niebla", 8: "otro", 9: "desconocido", -1: None},
    "condiciones_superficie": {1: "seca", 2: "mojada", 3: "nieve", 4: "hielo", 5: "inundada",
                               9: "desconocido", -1: None},
}


def colisiones():
    df = pd.read_csv(crudo("colisiones_2024.csv"), low_memory=False)
    assert df.shape == (100927, 44), f"la fuente de colisiones cambió: {df.shape}"
    df = df.rename(columns=COLISIONES)[list(COLISIONES.values())].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True)
    decodificar(df, COLISIONES_DECODIFICAR)
    assert df["distrito_ons"].notna().all() and df["distrito_ons"].nunique() > 300
    guardar(df, "colisiones_gb.parquet", (100927, 15))


# ── 6 · Netbilling (CNE) · solo tipos ────────────────────────────────────────────────────────────
def netbilling():
    df = pd.read_csv(crudo("netbilling.csv"), encoding="utf-8-sig")
    assert df.shape == (39636, 9), f"la fuente de Netbilling cambió: {df.shape}"
    df["potencia_kw"] = df["potencia_kw"].str.replace(",", ".", regex=False).astype(float)
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], dayfirst=True)
    df["fecha_act"] = pd.to_datetime(df["fecha_act"])
    guardar(df, "netbilling.parquet", (39636, 9))


if __name__ == "__main__":
    telemarketing()
    compras()
    trayectoria()
    bicicletas()
    colisiones()
    netbilling()
