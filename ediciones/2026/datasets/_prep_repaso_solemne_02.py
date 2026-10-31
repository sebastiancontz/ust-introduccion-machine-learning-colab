"""Genera los tres datasets sintéticos del repaso de la Unidad 2 (clases 5 a 8).

Son datos inventados para el curso: no representan a ninguna organización real. Cada archivo
esconde a propósito la decisión que su actividad ejercita:

- `servicios_mudanza.csv`: horas que toma una mudanza, con columnas que solo existen al TERMINAR
  el servicio (fuga de origen) y una numérica con vacíos, para que la mediana que la rellena se
  pueda ajustar bien o mal (fuga de procedimiento).
- `inasistencias_citas.csv`: si el paciente no se presenta a su cita; clase positiva minoritaria,
  para que la accuracy del baseline engañe y el umbral se decida por costos y no por F1.
- `cotizaciones_b2b.csv`: si una cotización se concreta en venta; con reglas de umbral, que es
  donde un árbol tiene algo que capturar y su profundidad algo que sobreajustar.

Los archivos de la Solemne 2 son otros y viven en `evaluaciones/datos/`: el repaso practica el
procedimiento sobre datos distintos para que no se memoricen cifras.

Uso: python ediciones/2026/datasets/_prep_repaso_solemne_02.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DESTINO = Path(__file__).resolve().parent
SEMILLA = 20261029


def servicios_mudanza(rng: np.random.Generator, n: int = 1500) -> pd.DataFrame:
    vivienda = rng.choice(["departamento", "casa"], n, p=[0.6, 0.4])
    ambientes = np.where(vivienda == "casa", rng.integers(2, 7, n), rng.integers(1, 5, n))
    metros = (ambientes * rng.normal(22, 4, n) + np.where(vivienda == "casa", 25, 8)).round(0)
    pisos = np.where(vivienda == "departamento",
                     np.where(rng.random(n) < 0.55, 0, rng.integers(1, 6, n)), 0)
    distancia = rng.gamma(2.0, 9.0, n).clip(1, 120).round(1)
    embalaje = rng.choice(["si", "no"], n, p=[0.35, 0.65])
    especiales = rng.choice(["si", "no"], n, p=[0.12, 0.88])
    region = rng.choice(["Metropolitana", "Valparaíso", "Biobío", "Araucanía", "Antofagasta"],
                        n, p=[0.5, 0.17, 0.14, 0.1, 0.09])

    horas = (
        1.0
        + 0.045 * metros
        + 0.45 * pisos
        + 0.035 * distancia
        + 1.4 * (embalaje == "si")
        + 1.1 * (especiales == "si")
    )
    horas = (horas * rng.lognormal(0, 0.22, n)).round(1).clip(1)

    # columnas que solo existen al TERMINAR el servicio
    cajas = (metros * 0.6 * rng.lognormal(0, 0.25, n)).round(0).astype(int)
    monto = (horas * 32_000 + distancia * 900 + rng.normal(0, 5_000, n)).round(-2).astype(int)
    reclamo = np.where(rng.random(n) < 0.04 + 0.01 * horas, "si", "no")

    # el metraje falta cuando el cliente no lo informó al cotizar
    metros_informados = metros.copy()
    metros_informados[rng.random(n) < 0.1] = np.nan

    return pd.DataFrame({
        "id_servicio": [f"MU-{i:05d}" for i in range(1, n + 1)],
        "region": region,
        "tipo_vivienda": vivienda,
        "ambientes": ambientes,
        "metros_cuadrados": metros_informados,
        "pisos_sin_ascensor": pisos,
        "distancia_km": distancia,
        "incluye_embalaje": embalaje,
        "objetos_especiales": especiales,
        "cajas_utilizadas": cajas,
        "monto_cobrado": monto,
        "reclamo_posterior": reclamo,
        "horas_servicio": horas,
    })


def inasistencias_citas(rng: np.random.Generator, n: int = 2500) -> pd.DataFrame:
    tipo = rng.choice(["control", "limpieza", "tratamiento", "evaluación"], n, p=[0.3, 0.25, 0.3, 0.15])
    canal = rng.choice(["web", "teléfono", "presencial"], n, p=[0.45, 0.35, 0.20])
    bloque = rng.choice(["mañana", "tarde", "sábado"], n, p=[0.45, 0.4, 0.15])
    anticipacion = rng.gamma(1.6, 7, n).clip(0, 60).round(0).astype(int)
    edad = rng.normal(40, 14, n).clip(18, 85).round(0).astype(int)
    previas = rng.poisson(3, n)
    faltas_previas = rng.binomial(previas, 0.12)
    recordatorio = rng.choice(["si", "no"], n, p=[0.6, 0.4])
    distancia = rng.gamma(2.0, 3.5, n).clip(0.3, 40).round(1)

    logit = (
        -2.3
        + 0.04 * anticipacion
        + 0.75 * faltas_previas
        - 0.8 * (recordatorio == "si")
        + 0.5 * (canal == "web")
        + 0.4 * (bloque == "sábado")
        - 0.5 * (tipo == "tratamiento")
        + 0.03 * distancia
        - 0.012 * (edad - 40)
    )
    falta = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        "id_cita": [f"CT-{i:05d}" for i in range(1, n + 1)],
        "tipo_atencion": tipo,
        "canal_agenda": canal,
        "bloque_horario": bloque,
        "dias_anticipacion": anticipacion,
        "edad_paciente": edad,
        "citas_previas": previas,
        "inasistencias_previas": faltas_previas,
        "recordatorio_enviado": recordatorio,
        "distancia_km": distancia,
        "no_se_presento": np.where(falta, "si", "no"),
    })


def cotizaciones_b2b(rng: np.random.Generator, n: int = 2000) -> pd.DataFrame:
    tamano = rng.choice(["pyme", "mediana", "grande"], n, p=[0.55, 0.3, 0.15])
    rubro = rng.choice(["construcción", "minería", "retail", "agrícola", "servicios"], n,
                       p=[0.3, 0.15, 0.25, 0.15, 0.15])
    recurrente = rng.choice(["si", "no"], n, p=[0.4, 0.6])
    canal = rng.choice(["vendedor", "web", "feria"], n, p=[0.5, 0.35, 0.15])
    escala = pd.Series({"pyme": 1.5, "mediana": 5.0, "grande": 15.0})[tamano].to_numpy()
    monto = (escala * 1_000_000 * rng.lognormal(0, 0.6, n)).round(-4).astype(int)
    descuento = rng.choice([0, 5, 10, 15], n, p=[0.4, 0.3, 0.2, 0.1])
    dias_respuesta = rng.integers(0, 11, n)
    productos = 1 + rng.poisson(3, n)

    # reglas de umbral: lo que una recta no ve y un árbol sí
    logit = (
        -1.2
        + 1.6 * (recurrente == "si")
        - 1.3 * (dias_respuesta > 3)
        + 0.8 * (descuento >= 10) * (recurrente == "no")
        - 0.7 * (monto > 8_000_000)
        + 0.5 * (canal == "vendedor")
    )
    cerrada = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        "id_cotizacion": [f"CZ-{i:05d}" for i in range(1, n + 1)],
        "rubro_cliente": rubro,
        "tamano_cliente": tamano,
        "es_cliente_recurrente": recurrente,
        "canal_origen": canal,
        "monto_cotizado": monto,
        "descuento_pct": descuento,
        "dias_respuesta": dias_respuesta,
        "cantidad_productos": productos,
        "se_concreto": np.where(cerrada, "si", "no"),
    })


if __name__ == "__main__":
    rng = np.random.default_rng(SEMILLA)
    archivos = {
        "servicios_mudanza.csv": servicios_mudanza(rng),
        "inasistencias_citas.csv": inasistencias_citas(rng),
        "cotizaciones_b2b.csv": cotizaciones_b2b(rng),
    }
    for nombre, datos in archivos.items():
        datos.to_csv(DESTINO / nombre, index=False)
        print(f"{nombre}: {datos.shape[0]} registros x {datos.shape[1]} columnas")
