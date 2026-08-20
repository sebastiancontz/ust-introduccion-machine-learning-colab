"""Prepara el dataset de ventas de la clase 4 (UCI 502, hoja 2010-2011) -> Parquet en espanol.

Autocontenido: descarga el original desde UCI, renombra las columnas al espanol y escribe el
Parquet.  Ejecutar desde cualquier directorio:  python _prep_ventas_online.py

EL ARCHIVO NO SE LIMPIA. Es el insumo de una clase de PREPARACION DE DATOS: los defectos son el
contenido, no un accidente. Lo unico que se toca son los NOMBRES de columna. Los VALORES llegan
crudos, incluidos los que estan en ingles: traducirlos convertiria un dato real en uno fabricado y
ademas borraria defectos (`EIRE` y `RSA` conviven con nombres de pais completos, y `Unspecified` es
un faltante disfrazado de categoria).

PARQUET Y NO CSV: el tipo de cada columna viaja guardado, asi que el archivo no depende de que
quien lo lea acierte con los argumentos de lectura. Pesa 3 MB contra 48 MB del CSV equivalente.
Efecto lateral que conviene tener presente al ensenar: `id_cliente` queda congelado como decimal
—que es justo el defecto de tipo que la clase discute— en vez de ser un artefacto de la lectura.

FALLA CERRADO: comprueba con `assert` cada defecto que el material de la clase afirma y, si alguno
no se cumple, NO escribe el archivo y termina con codigo de error. Es preferible quedarse sin
dataset que publicar uno que contradiga las slides. Las cifras se verificaron el 2026-08-04 y son
las que estan escritas en la `calibracion` de la clase 4 en `temario.yml`.
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
HOJA = "Year 2010-2011"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ventas_online.parquet")

RENOMBRE = {
    "Invoice": "n_factura",
    "StockCode": "codigo_producto",
    "Description": "descripcion_producto",
    "Quantity": "cantidad",
    "InvoiceDate": "fecha_factura",
    "Price": "precio_unitario",
    "Customer ID": "id_cliente",
    "Country": "pais",
}


def descargar():
    with urllib.request.urlopen(URL) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    nombre = next(n for n in z.namelist() if n.endswith(".xlsx"))
    # `Description` se fuerza a texto porque UNA de sus celdas guarda un numero (un codigo de
    # producto escrito en el campo del nombre). Sin esto la columna queda de tipo mixto y Parquet
    # no la puede escribir. El valor se conserva tal cual: el defecto es el contenido, no el tipo.
    return pd.read_excel(z.open(nombre), sheet_name=HOJA, engine="openpyxl",
                         dtype={"Invoice": str, "StockCode": str, "Description": str})


df = descargar().rename(columns=RENOMBRE)[list(RENOMBRE.values())]

# ── Verificacion: cada defecto que la clase 4 ensena tiene que estar en el archivo ─────────────
assert df.shape == (541910, 8), f"forma inesperada: {df.shape}"
assert list(df.columns) == list(RENOMBRE.values()), "faltan o sobran columnas"

# Concepto 1 — formato y tipo
assert df["id_cliente"].dtype == "float64", "id_cliente ya no llega como decimal"
cancel = df["n_factura"].str.startswith("C")
assert cancel.sum() == 9288, f"lineas canceladas: {cancel.sum()}"
noprod = df.loc[~df["codigo_producto"].str.match(r"^\d{5}"), "codigo_producto"]
for cod in ("POST", "DOT", "M", "D", "BANK CHARGES", "AMAZONFEE", "CRUK"):
    assert cod in noprod.values, f"desaparecio el codigo que no es un producto: {cod}"
multi = (df.groupby("codigo_producto")["descripcion_producto"].nunique() > 1).sum()
assert multi == 650, f"codigos con mas de una descripcion: {multi}"
for anot in ("check", "damages", "?", "found", "wrongly coded 23343"):
    assert (df["descripcion_producto"] == anot).any(), f"desaparecio la anotacion a mano: {anot}"
# El mismo incidente anotado desde los dos lados: en la fila del codigo 23343 alguien escribio el
# OTRO codigo en el campo del nombre, y las filas del codigo 20713 lo explican con palabras.
cruce = df[(df["codigo_producto"] == "23343") & (df["descripcion_producto"] == "20713")]
assert len(cruce) == 1, f"se perdio el codigo escrito en el campo del nombre: {len(cruce)} filas"
for cat in ("Unspecified", "European Community"):
    assert (df["pais"] == cat).any(), f"desaparecio la categoria de pais: {cat}"

# Concepto 2 — faltantes y su mecanismo
sin_cliente = df["id_cliente"].isna()
assert sin_cliente.sum() == 135080, f"id_cliente faltante: {sin_cliente.sum()}"
assert df["descripcion_producto"].isna().sum() == 1454, "cambio el faltante de descripcion"
# Ausencia informativa: las filas sin cliente NO se comportan como las demas.
media_con = df.loc[~sin_cliente, "cantidad"].mean()
media_sin = df.loc[sin_cliente, "cantidad"].mean()
assert media_sin < media_con / 3, (
    f"la ausencia dejo de ser informativa: cantidad media {media_sin:.1f} sin cliente "
    f"contra {media_con:.1f} con cliente")

# Concepto 4 — duplicados
assert df.duplicated().sum() == 5268, f"duplicados exactos: {df.duplicated().sum()}"

# Concepto 5 — atipicos: el par real que NO se borra, y el ajuste que no es una venta
par = df[df["n_factura"].isin(("581483", "C581484"))]
assert sorted(par["cantidad"]) == [-80995, 80995], "se perdio el pedido mas grande y su devolucion"
assert par["id_cliente"].nunique() == 1, "el par dejo de ser del mismo cliente"
assert (df["descripcion_producto"] == "printing smudges/thrown away").any(), \
    "se perdio el ajuste de inventario cargado en la tabla de ventas"
# Coherencia interna: toda linea cancelada es negativa, pero hay negativas sin cancelar.
assert (df.loc[cancel, "cantidad"] > 0).sum() == 0, "hay lineas canceladas con cantidad positiva"
sueltas = ((~cancel) & (df["cantidad"] < 0)).sum()
assert sueltas == 1336, f"negativas sin prefijo de cancelacion: {sueltas}"
assert (df["precio_unitario"] == 0).sum() == 2515, "cambio el conteo de precio cero"

df.to_parquet(OUT, compression="zstd", index=False)

# ── Resumen ────────────────────────────────────────────────────────────────────────────────────
print(df.shape, "| facturas:", df["n_factura"].nunique(), "| clientes:", int(df["id_cliente"].nunique()))
print("periodo:", df["fecha_factura"].min(), "->", df["fecha_factura"].max())
print("faltantes:", df.isna().sum().sum(), "| duplicados exactos:", df.duplicated().sum())
print("id_cliente faltante: %d (%.1f %%)" % (sin_cliente.sum(), 100 * sin_cliente.mean()))
print("cantidad media  con cliente: %.1f  |  sin cliente: %.1f" % (media_con, media_sin))
print("cantidad: min %d max %d" % (df["cantidad"].min(), df["cantidad"].max()))
print("paises:", df["pais"].nunique(), "| codigos de producto:", df["codigo_producto"].nunique())
print("\nbytes:", os.path.getsize(OUT))
print(df.dtypes.to_string())
