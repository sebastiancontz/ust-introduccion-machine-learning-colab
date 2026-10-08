"""Genera `cobranza_servicios.csv`, el archivo de la Clase 3 (análisis exploratorio y calidad).

POR QUÉ EXISTE ESTE ARCHIVO
---------------------------
La clase 3 se armó primero sobre `absentismo_laboral.csv`, el caso de la clase 2, y no
funcionó: cada hallazgo exigía contexto que no estaba en los datos (estaciones del hemisferio
sur, cifras de planta, códigos CIE-10, un cero que a veces no era una ausencia). Para una clase
de AUDITORÍA eso está al revés — el defecto tiene que descubrirse con una técnica, no porque el
docente explicó antes el dominio.

Este archivo es sintético y está hecho para que cada concepto de la clase tenga UN defecto
evidente, en un dominio que la audiencia ya maneja: facturación y cobranza. Es además el caso
que los propios estudiantes formularon en la práctica autónoma de la clase 2, así que la
continuidad se mantiene.

DATOS SINTÉTICOS, y hay que decirlo en clase: no son de ninguna empresa real. Los RUT son
inventados con dígito verificador válido para que se vean creíbles.

Determinista: semilla fija, sin fechas del sistema. Correr dos veces da el mismo archivo.

Uso:  conda run -n ust-ml python ediciones/2026/datasets/_prep_cobranza_servicios.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SALIDA = Path(__file__).resolve().parent / "cobranza_servicios.csv"
SEMILLA = 42
N = 600

rng = np.random.default_rng(SEMILLA)

SEGMENTOS = ["pyme", "corporativo", "gobierno"]
REGIONES = ["Metropolitana", "Valparaíso", "Biobío", "Antofagasta", "Los Lagos"]
EJECUTIVOS = ["A. Rojas", "M. Fuentes", "C. Peña", "J. Salas", "P. Vergara"]
PRIORIDADES = ["baja", "media", "alta"]          # ordinal
DIAS_CREDITO = [30, 60, 90]                      # numérica que en realidad es categórica


def digito_verificador(cuerpo: int) -> str:
    suma, factor = 0, 2
    for d in reversed(str(cuerpo)):
        suma += int(d) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - suma % 11
    return {11: "0", 10: "K"}.get(resto, str(resto))


def rut(cuerpo: int) -> str:
    return f"{cuerpo:,}".replace(",", ".") + "-" + digito_verificador(cuerpo)


def construir():
    # ── cartera de clientes ────────────────────────────────────────────────
    n_clientes = 80
    cuerpos = rng.choice(np.arange(60_000_000, 79_999_999), size=n_clientes, replace=False)
    ruts = [rut(int(c)) for c in cuerpos]
    nombres = [f"{p} {s}"
               for p, s in zip(rng.choice(["Comercial", "Servicios", "Distribuidora", "Ingeniería",
                                           "Transportes", "Constructora"], n_clientes),
                               rng.choice(["Andes", "del Sur", "Pacífico", "Central", "Aconcagua",
                                           "Araucanía", "Norte Grande", "Cordillera"], n_clientes))]
    seg_cliente = rng.choice(SEGMENTOS, n_clientes, p=[0.6, 0.3, 0.1])
    reg_cliente = rng.choice(REGIONES, n_clientes, p=[0.45, 0.2, 0.15, 0.1, 0.1])
    eje_cliente = rng.choice(EJECUTIVOS, n_clientes)

    idx = rng.integers(0, n_clientes, N)

    # ── fechas: 12 meses corridos ──────────────────────────────────────────
    inicio = pd.Timestamp("2025-01-02")
    emision = inicio + pd.to_timedelta(rng.integers(0, 364, N), unit="D")
    dias_credito = rng.choice(DIAS_CREDITO, N, p=[0.6, 0.3, 0.1])
    vencimiento = emision + pd.to_timedelta(dias_credito, unit="D")

    # ── monto: asimétrico, que es lo que pide el concepto 2 ───────────────
    monto = np.round(rng.lognormal(mean=13.4, sigma=0.75, size=N), -3).astype(int)

    # ── objetivo: días de atraso ──────────────────────────────────────────
    # Mezcla realista: una parte de la cartera paga al día y el resto se atrasa con cola
    # larga. El segmento mueve las dos cosas — el gobierno paga tarde y casi nunca al día.
    p_al_dia = np.where(seg_cliente[idx] == "corporativo", 0.55,
                np.where(seg_cliente[idx] == "pyme", 0.32, 0.10))
    base = np.where(seg_cliente[idx] == "pyme", 12, np.where(seg_cliente[idx] == "gobierno", 26, 6))
    atraso = np.where(rng.random(N) < p_al_dia, 0,
                      np.clip(rng.poisson(base) + rng.integers(-4, 9, N), 1, None))

    d = pd.DataFrame({
        "id_factura": [f"F-{i:04d}" for i in range(1, N + 1)],
        "rut_cliente": [ruts[i] for i in idx],
        "razon_social": [nombres[i] for i in idx],
        "segmento": seg_cliente[idx],
        "region": reg_cliente[idx],
        "ejecutivo": eje_cliente[idx],
        "prioridad_cobranza": rng.choice(PRIORIDADES, N, p=[0.5, 0.35, 0.15]),
        "fecha_emision": emision,
        "fecha_vencimiento": vencimiento,
        "dias_credito": dias_credito,
        "monto_neto": monto,
        "dias_atraso": atraso,
    })
    d["fecha_pago"] = d.fecha_vencimiento + pd.to_timedelta(d.dias_atraso, unit="D")

    # ── redundancia perfecta: el IVA (concepto 3) ─────────────────────────
    d["monto_total"] = (d.monto_neto * 1.19).round().astype(int)

    # ── señales de fuga (concepto 7) ──────────────────────────────────────
    # 1. se conoce después del hecho
    d["estado_cobranza"] = np.where(d.dias_atraso == 0, "pagada al día",
                             np.where(d.dias_atraso <= 30, "pagada con atraso", "en gestión judicial"))
    # 2. casi determina la respuesta: solo existe si hubo atraso
    d["monto_gestion_cobranza"] = np.where(d.dias_atraso > 0,
                                           (d.monto_neto * 0.02).round().astype(int), 0)
    # 3. agregado del período: la morosidad de TODA la cartera ese mes
    mes = d.fecha_emision.dt.to_period("M")
    tasa_mes = {m: round(v, 1) for m, v in zip(sorted(mes.unique()),
                                               rng.uniform(2.5, 7.5, mes.nunique()))}
    # llega como TEXTO con coma decimal, que es como sale de casi todo sistema chileno
    d["tasa_morosidad_cartera"] = mes.map(tasa_mes).astype(str).str.replace(".", ",", regex=False)

    d = d[["id_factura", "rut_cliente", "razon_social", "segmento", "region", "ejecutivo",
           "prioridad_cobranza", "fecha_emision", "fecha_vencimiento", "fecha_pago",
           "dias_credito", "monto_neto", "monto_total", "tasa_morosidad_cartera",
           "estado_cobranza", "monto_gestion_cobranza", "dias_atraso"]]

    # ══════════════════════════════════════════════════════════════════════
    # DEFECTOS. Cada uno alimenta un concepto y se cuenta exacto.
    # ══════════════════════════════════════════════════════════════════════
    culpable = "M. Fuentes"          # los defectos se concentran en un ejecutivo (hallazgo de "rebanadas")

    # (a) un contrato grande REAL: atípico que no es error (concepto 6)
    d.loc[d.index[7], ["monto_neto", "segmento", "razon_social"]] = [148_000_000, "gobierno",
                                                                     "Constructora Cordillera"]
    d.loc[d.index[7], "monto_total"] = round(148_000_000 * 1.19)

    # (b) 8 montos cargados en MILES: atípico que sí es error (concepto 6)
    en_miles = d.index[[31, 88, 145, 203, 268, 341, 402, 477]]
    d.loc[en_miles, "monto_neto"] = (d.loc[en_miles, "monto_neto"] / 1000).round().astype(int)
    d.loc[en_miles, "monto_total"] = (d.loc[en_miles, "monto_neto"] * 1.19).round().astype(int)

    # (c) 5 facturas pagadas ANTES de emitirse: imposible lógico (concepto 5)
    imposibles = d.index[[12, 96, 210, 333, 455]]
    d.loc[imposibles, "fecha_pago"] = d.loc[imposibles, "fecha_emision"] - pd.Timedelta(days=6)
    d.loc[imposibles, "ejecutivo"] = culpable

    # (d) 4 RUT con más de una razón social (concepto 5)
    for pos, variante in zip([20, 57, 130, 260],
                             [str.upper, lambda s: s + "  S.A.", lambda s: s.replace(" ", "  "),
                              lambda s: "Comercial Andes Limitada"]):
        r = d.loc[d.index[pos], "rut_cliente"]
        filas = d.index[(d.rut_cliente == r)][:2]
        if len(filas) == 2:
            d.loc[filas[1], "razon_social"] = variante(d.loc[filas[1], "razon_social"])

    # (e) 12 duplicados exactos, concentrados en el mismo ejecutivo (concepto 5)
    d.loc[d.index[:40], "ejecutivo"] = np.where(rng.random(40) < 0.5, culpable,
                                                d.loc[d.index[:40], "ejecutivo"])
    # ni las filas imposibles ni el contrato grande entran como fuente de duplicados:
    # cada defecto tiene que poder contarse por separado
    excluir = list(imposibles) + [d.index[7]] + list(en_miles)
    fuente = d[(d.ejecutivo == culpable) & (~d.index.isin(excluir))].index[:12]
    d = pd.concat([d, d.loc[fuente]], ignore_index=True)

    d = d.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    return d


def verificar(d):
    """Cada afirmación que la clase hace sobre este archivo se comprueba acá.

    El `__main__` falla cerrado si alguna no se cumple: mejor no escribir el CSV que publicar
    uno que contradiga el material.
    """
    ok = []
    dup = int(d.duplicated().sum())
    ruts_multi = int((d.groupby("rut_cliente").razon_social.nunique() > 1).sum())
    imposibles = int((pd.to_datetime(d.fecha_pago) < pd.to_datetime(d.fecha_emision)).sum())
    corr_iva = d.monto_neto.corr(d.monto_total)
    tasa_por_mes = pd.to_datetime(d.fecha_emision).dt.to_period("M")
    constante = int((d.groupby(tasa_por_mes).tasa_morosidad_cartera.nunique() == 1).all())
    gestion_cero = float((d.loc[d.dias_atraso == 0, "monto_gestion_cobranza"] == 0).mean())

    ok.append(("filas", len(d), len(d) == 612))
    ok.append(("el contrato grande NO está duplicado", int((d.monto_neto > 20e6).sum()), int((d.monto_neto > 20e6).sum()) == 1))
    ok.append(("duplicados exactos", dup, dup == 12))
    ok.append(("RUT con >1 razón social", ruts_multi, ruts_multi >= 4))  # 4 inyectados; puede subir si dos caen en el mismo RUT
    ok.append(("pago anterior a emisión", imposibles, imposibles == 5))
    ok.append(("corr monto_neto/monto_total", round(corr_iva, 4), corr_iva > 0.9999))
    ok.append(("tasa constante dentro del mes", constante, constante == 1))
    ok.append(("gestión = 0 si no hubo atraso", f"{gestion_cero:.0%}", gestion_cero == 1.0))
    es_texto = not pd.api.types.is_numeric_dtype(d.tasa_morosidad_cartera)
    ok.append(("tasa llega como texto", d.tasa_morosidad_cartera.dtype, es_texto))
    ok.append(("monto: media > mediana", f"{d.monto_neto.mean():,.0f} > {d.monto_neto.median():,.0f}",
               d.monto_neto.mean() > d.monto_neto.median()))
    return ok


if __name__ == "__main__":
    import sys

    d = construir()
    resultados = verificar(d)
    for nombre, valor, bien in resultados:
        print(f"  {'OK  ' if bien else 'FALLA'}  {nombre:32s} {valor}")

    fallas = [n for n, _, bien in resultados if not bien]
    if fallas:
        # FALLA CERRADO a propósito: la clase 3 afirma cada uno de estos hechos sobre el archivo.
        # Si alguno deja de cumplirse, el CSV no se escribe — antes que publicar un dataset que
        # contradiga el material.
        print(f"\nNO se escribió {SALIDA.name}: fallaron {len(fallas)} verificaciones -> {fallas}")
        sys.exit(1)

    d.to_csv(SALIDA, index=False)
    print(f"\nescrito {SALIDA.name}: {d.shape[0]} filas × {d.shape[1]} columnas")
