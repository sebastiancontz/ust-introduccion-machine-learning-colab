"""Deriva la tabla POR CLIENTE de la clase 7 desde `ventas_online_limpio.parquet`.

POR QUE EXISTE. La clase 7 abre la Unidad 2 con el primer problema predictivo BIEN FORMULADO del
curso: cuantas veces vuelve a comprar un cliente. Las clases 4 a 6 trabajaron el mismo archivo con
otras dos unidades de observacion -la LINEA de factura y la FACTURA-, y su objetivo -el monto de la
propia factura- era un instrumento de medicion, no una decision. Aca la unidad es el CLIENTE y hay
horizonte temporal de verdad.

LA REGLA QUE ORDENA TODO EL ARCHIVO: hay un CORTE. Toda columna predictora se calcula solo con lo
ocurrido ANTES del corte; el objetivo, solo con lo ocurrido DESPUES. El sufijo `_base` en el nombre
de una columna no es decorativo: dice que ese dato estaba disponible en el momento de predecir. Es la
leccion de la clase 2 -momento de la prediccion- y la de la clase 6 -de donde viene cada variable-
aplicadas al diseno del archivo y no a una advertencia.

LA POBLACION NO SE FILTRA POR EL OBJETIVO. Un borrador anterior recortaba el 1 % superior del gasto
futuro para estabilizar los coeficientes: eso es seleccionar por el resultado que no esta disponible
al predecir, o sea la misma falta que la clase 6 ensena a detectar. Se descarto. Tampoco se filtra por
pais: los 36 paises se agrupan en `mercado`, que conserva las cifras y le da a la clase una predictora
categorica con categoria de referencia.

UNICA DECISION DECLARADA: el cliente que no vuelve a comprar en la ventana futura vale CERO, no
faltante. Es una observacion valida -no volvio- y son 1.367 de 3.314. Misma decision que la grilla de
empleado-mes de la clase 2, y por eso la mediana del objetivo es 1 mientras el promedio es 1,63.

EL IDENTIFICADOR SE CONVIERTE A ENTERO, y se declara: en el archivo de origen `id_cliente` llega
como decimal, que es el defecto de tipo que la clase 4 discute a proposito. Esa leccion ya ocurrio; en
una tabla por cliente un identificador con coma es ruido, asi que aca se convierte a entero. Es la
unica transformacion de tipo del script.

DETERMINISTA: sin semillas, sin fechas del sistema, sin red. El corte esta escrito en el codigo.
Correrlo dos veces produce el mismo archivo byte a byte.

FALLA CERRADO: comprueba con `assert` cada cifra que el material de la clase afirma y, si alguna no se
cumple, NO escribe el archivo y termina con codigo de error. Es preferible quedarse sin dataset que
publicar uno que contradiga las slides.

Uso:  conda run -n ust-ml python ediciones/2026/datasets/_prep_clientes_ventas.py
"""
import os

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
ORIGEN = os.path.join(AQUI, "ventas_online_limpio.parquet")
OUT = os.path.join(AQUI, "clientes_ventas.parquet")

# El corte entre las dos ventanas. Nueve meses de historia, tres meses de horizonte.
CORTE = pd.Timestamp("2011-09-01")


def base_analitica(origen: str) -> pd.DataFrame:
    """El MISMO filtro de las clases 5 y 6, para que la base no se vuelva a discutir."""
    df = pd.read_parquet(origen)
    vivo = (
        df["es_producto"]
        & ~df["cancelada"]
        & ~df["sin_cliente"]
        & (df["cantidad"] > 0)
        & (df["precio_unitario"] > 0)
    )
    df = df.loc[vivo].copy()
    df["monto"] = df["cantidad"] * df["precio_unitario"]
    return df


def construir(df: pd.DataFrame) -> pd.DataFrame:
    antes = df[df["fecha_factura"] < CORTE]
    despues = df[df["fecha_factura"] >= CORTE]

    # Predictoras: SOLO con la ventana base. El sufijo lo declara en el nombre.
    g = antes.groupby("id_cliente")
    tabla = pd.DataFrame(
        {
            "facturas_base": g["n_factura"].nunique(),
            "monto_base": g["monto"].sum().round(2),
            "productos_base": g["codigo_producto"].nunique(),
            "recencia_dias": (CORTE - g["fecha_factura"].max()).dt.days,
            "antiguedad_dias": (CORTE - g["fecha_factura"].min()).dt.days,
            "pais": g["pais"].first(),
        }
    )
    tabla["ticket_medio_base"] = (tabla["monto_base"] / tabla["facturas_base"]).round(2)
    # 36 paises agrupados en dos: la categorica de la clase, con su categoria de referencia.
    tabla["mercado"] = tabla["pais"].where(
        tabla["pais"] == "United Kingdom", "exportacion"
    ).replace({"United Kingdom": "reino unido"})

    # Objetivo: SOLO con la ventana futura. El que no vuelve vale cero, no faltante.
    objetivo = (
        despues.groupby("id_cliente")["n_factura"].nunique().reindex(tabla.index, fill_value=0)
    )
    tabla["facturas_futuras"] = objetivo.astype("int64")

    # El identificador a entero: la leccion del tipo decimal es de la clase 4, no de esta.
    tabla.index = tabla.index.astype("int64")

    orden = [
        "facturas_base", "monto_base", "productos_base", "ticket_medio_base",
        "recencia_dias", "antiguedad_dias", "mercado", "facturas_futuras",
    ]
    return tabla[orden].reset_index().rename(columns={"index": "id_cliente"}).sort_values("id_cliente")


df = base_analitica(ORIGEN)
tabla = construir(df)

# ── Falla cerrado: cada cifra que el material afirma ───────────────────────────────────────────
assert len(tabla) == 3314, f"clientes: {len(tabla)}"
assert tabla["id_cliente"].nunique() == len(tabla), "id_cliente no identifica al cliente"
assert tabla["id_cliente"].dtype == "int64", f"id_cliente quedo como {tabla['id_cliente'].dtype}"
assert tabla.isna().sum().sum() == 0, "la tabla no puede tener faltantes: cada columna se construye"

# La unica decision declarada, y su consecuencia sobre la forma del objetivo.
ceros = int((tabla["facturas_futuras"] == 0).sum())
assert ceros == 1367, f"clientes que no vuelven: {ceros}"
assert abs(tabla["facturas_futuras"].mean() - 1.63) < 0.01, "cambio el promedio del objetivo"
assert tabla["facturas_futuras"].median() == 1, "cambio la mediana del objetivo"
assert tabla["facturas_futuras"].max() == 95, f"maximo del objetivo: {tabla['facturas_futuras'].max()}"
assert (tabla["facturas_futuras"] >= 0).all(), "un conteo no puede ser negativo"

# La categorica y su reparto. `exportacion` es la categoria de referencia de la clase.
reparto = tabla["mercado"].value_counts()
assert set(reparto.index) == {"reino unido", "exportacion"}, reparto.index.tolist()
assert reparto["reino unido"] == 2986 and reparto["exportacion"] == 328, reparto.to_dict()

# LA REGLA DEL CORTE, comprobada y no solo prometida: ninguna predictora puede haber mirado
# despues del corte. Se verifica reconstruyendo la tabla con las lineas posteriores BORRADAS:
# las predictoras tienen que salir identicas, y el objetivo entero en cero.
solo_antes = construir(df[df["fecha_factura"] < CORTE])
predictoras = [c for c in tabla.columns if c != "facturas_futuras"]
pd.testing.assert_frame_equal(
    tabla[predictoras].reset_index(drop=True), solo_antes[predictoras].reset_index(drop=True)
)
assert (solo_antes["facturas_futuras"] == 0).all(), \
    "el objetivo se calculo con datos anteriores al corte"

# Y al reves: el objetivo NO puede depender de la ventana base. Con las lineas anteriores
# borradas cambia la poblacion, pero el objetivo de los clientes que sobreviven es el mismo.
solo_despues = df[df["fecha_factura"] >= CORTE].groupby("id_cliente")["n_factura"].nunique()
comun = tabla.set_index("id_cliente")["facturas_futuras"]
comun = comun[comun > 0]
assert (comun == solo_despues.reindex(comun.index)).all(), "el objetivo no sale solo de la ventana futura"

tabla.to_parquet(OUT, compression="zstd", index=False)

# ── Resumen ───────────────────────────────────────────────────────────────────────────────────
print(tabla.shape, "| clientes:", tabla["id_cliente"].nunique())
print("corte:", CORTE.date(),
      "| ventana base:", df.loc[df["fecha_factura"] < CORTE, "fecha_factura"].min().date(),
      "->", df.loc[df["fecha_factura"] < CORTE, "fecha_factura"].max().date(),
      "| ventana futura:", df.loc[df["fecha_factura"] >= CORTE, "fecha_factura"].min().date(),
      "->", df.loc[df["fecha_factura"] >= CORTE, "fecha_factura"].max().date())
print("objetivo facturas_futuras: media %.2f | mediana %.0f | max %d | en cero %d (%.0f %%)"
      % (tabla["facturas_futuras"].mean(), tabla["facturas_futuras"].median(),
         tabla["facturas_futuras"].max(), ceros, 100 * ceros / len(tabla)))
print("mercado:", reparto.to_dict())
print("correlacion de cada predictora con el objetivo:")
print(tabla.drop(columns=["id_cliente", "mercado"]).corr()["facturas_futuras"].round(3).to_string())
print("\nbytes:", os.path.getsize(OUT))
print(tabla.dtypes.to_string())
