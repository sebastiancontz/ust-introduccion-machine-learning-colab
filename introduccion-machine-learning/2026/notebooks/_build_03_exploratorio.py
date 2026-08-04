#!/usr/bin/env python3
"""Construye notebooks/03-exploratorio.ipynb.

POR QUÉ EXISTE UN GENERADOR: en la clase 1 el notebook se armó con un script que quedó en un
scratchpad temporal, y al regenerarlo se revirtieron en silencio dos correcciones que solo estaban
parchadas en el .ipynb. Regla del repo: el generador se versiona JUNTO al artefacto y el .ipynb no
se edita a mano. Si hay que corregir algo, se corrige acá y se vuelve a ejecutar.

ESTRUCTURA: es la rutina de auditoría de cinco pasos de las slides, en el mismo orden, con la
liberación gradual del bloque práctico —pasos 1 y 2 los muestra el docente, 3 y 4 se hacen en
conjunto, el 5 lo trabajan por su cuenta—. El resultado del notebook es una BITÁCORA de hallazgos,
no un archivo corregido: corregir es la clase 4.

CIFRAS: todas verificadas contra `datasets/cobranza_servicios.csv` el 2026-08-03. Si el dataset se
regenera, hay que volver a verificarlas, porque acá se afirman en el texto.

COMPATIBILIDAD: nada del código depende del nombre literal de un dtype. pandas 3 informa `str`
donde pandas 2 informa `object`, y Colab no siempre trae la misma versión que el entorno local; los
tipos se consultan con `select_dtypes`, que se comporta igual en ambas.

Uso:  python introduccion-machine-learning/2026/notebooks/_build_03_exploratorio.py
      (después: ejecutar el notebook para dejarlo con salidas, ver el README del repo)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "03-exploratorio.ipynb"
SVG = REPO_ROOT / "assets" / "ilustraciones" / "rutina-auditoria.svg"


def figura_base64(path: Path, alt: str, width: int = 900) -> str:
    """Diagrama embebido como data URI: el notebook queda autocontenido y no depende de Pages.

    `alt` es obligatorio: sin él la figura no existe para quien usa lector de pantalla, y
    `nbconvert` lo reclama al exportar.
    """
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<p align="center"><img src="data:image/svg+xml;base64,{b64}" '
            f'width="{width}" alt="{alt}"></p>')


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip("\n").splitlines(keepends=True)}


def code(texto: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": texto.strip("\n").splitlines(keepends=True)}


def con_ids(celdas: list[dict]) -> list[dict]:
    """nbformat 4.5+ exige un id por celda. Deterministas: regenerar no produce ruido en el diff."""
    for i, celda in enumerate(celdas):
        celda["id"] = f"c{i:02d}"
    return celdas


CELDAS = [
    md("""
<div style="display:flex; align-items:center; gap:18px; text-align:left">
<img src="https://sebastiancontz.github.io/ust-introduccion-machine-learning/assets/logo-ust.svg" width="100" alt="Logo de la Universidad Santo Tomás">
<div>
<p>Ingeniería en Información y Control de Gestión</p>
<p>Facultad de Economía y Negocios</p>
<p>Introducción a Machine Learning</p>
<p>Semana 03: Análisis exploratorio y calidad de datos</p>
</div>
</div>
"""),
    md("""
# 03 · Análisis exploratorio y calidad de datos

Una cartera de **612 facturas** de una empresa de servicios. El archivo declara **cero celdas
vacías**, y aun así no está limpio.

Hoy no se corrige nada. Se **detecta** y se **documenta**. Lo que sale de este notebook es una
**bitácora de hallazgos**, que es el insumo de la clase 4.

> Los datos son **sintéticos**, creados para el curso. El caso es real en su estructura, pero las
> cifras no sirven como referencia de mercado.
"""),
    md("""
## La rutina

Los siete conceptos de la clase, convertidos en un procedimiento de cinco pasos. Este notebook es
esa rutina, en ese orden.
"""),
    md(figura_base64(
        SVG, width=980,
        alt="Rutina de auditoría en cinco pasos numerados. Paso uno, qué es cada columna: tipos y "
            "cuáles no son predictoras, con info y dtypes, y caza un número que llegó como texto. "
            "Paso dos, cómo se comporta cada una: cuántos datos hay y su centro, dispersión y forma, "
            "con describe y el histograma, y caza un promedio que esconde la forma. Paso tres, cómo "
            "se relacionan entre sí, de a dos y todas a la vez, con corr y groupby, y caza dos "
            "columnas que dicen lo mismo. Paso cuatro, qué se contradice: lo que debería cumplirse "
            "siempre, con duplicated y nunique, y caza un RUT con dos razones sociales. Paso cinco, "
            "de qué momento es cada dato y si se conoce antes de decidir, con crosstab y desviación "
            "estándar, y caza una cifra que solo existe después. Al pie, el resultado es una "
            "bitácora de hallazgos y no un archivo corregido")),
    md("""
El orden no es arbitrario: no se puede describir una columna sin saber **qué tipo** es, ni buscar
contradicciones entre columnas sin conocer **cada una** por separado.
"""),
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas, así que la celda siguiente termina en segundos. Está
igual porque el notebook también tiene que ejecutarse fuera de Colab, y porque dejar escrito de qué
depende un análisis es parte de que sea reproducible.
"""),
    code("""
%%capture
!pip install -q pandas matplotlib
"""),
    md("""
## Cargar datos

En **Colab** los datos se leen del repo público del curso; en local, desde `../datasets/`. La misma
celda funciona en los dos casos.

Se lee **sin indicarle nada a pandas** sobre los tipos. Eso es a propósito: parte de lo que hay que
auditar es qué decidió pandas por su cuenta.
"""),
    code("""
import os
import pandas as pd

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/introduccion-machine-learning/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

facturas = pd.read_csv(BASE + 'cobranza_servicios.csv')

facturas.info()

print()
print('Celdas vacías:', facturas.isna().sum().sum())
"""),
    md("""
## La bitácora

Un hallazgo sin evidencia no es un hallazgo. «La columna de montos tiene problemas» no dice cuál, no
lo demuestra y no dice a quién le importa.

Cada hallazgo se registra con seis campos. La función de abajo los pide todos.
"""),
    code("""
BITACORA = []

def anotar(variable, categoria, descripcion, evidencia, severidad, impacto):
    \"\"\"Agrega una línea a la bitácora y la devuelve para revisarla.

    categoria : 'formato' | 'cantidad' | 'distribución' | 'momento'
    severidad : 'alta' | 'media' | 'baja'
    evidencia : la salida concreta que lo demuestra, no una impresión
    impacto   : a qué decisión afecta. Si no afecta a ninguna, revisar si es un hallazgo.
    \"\"\"
    BITACORA.append({'variable': variable, 'categoría': categoria, 'descripción': descripcion,
                     'evidencia': evidencia, 'severidad': severidad, 'impacto': impacto})
    return pd.DataFrame(BITACORA[-1:])
"""),
    md("""
---

# Parte 1 · Miren esto conmigo

Los pasos 1 y 2 de la rutina, con el ciclo completo: pregunta, mirada, hallazgo, línea de bitácora.
"""),
    md("""
## Paso 1 · ¿Qué es cada columna?

Antes de calcular nada hay que saber con qué se está tratando. Diecisiete columnas.

> **Antes de ejecutar:** el archivo tiene columnas de identificación, de texto, de fecha y de
> número. ¿Cuántas creen que pandas leyó como **número**?

Según la versión de pandas, las columnas de texto aparecen como `object` o como `str`. Significan lo
mismo: **no es un número**.
"""),
    code("""
facturas.dtypes
"""),
    md("""
Ahora la misma pregunta, pero contada por `describe()`, que es lo que casi todo el mundo ejecuta
primero.
"""),
    code("""
facturas.describe()
"""),
    md("""
> **Cuenten las columnas de la salida anterior.** Después miren esta lista de nombres y digan cuál
> de todas ellas *debería* estar ahí y no está.
"""),
    code("""
# Las que pandas trata como números
numericas = facturas.select_dtypes(include='number').columns.tolist()

# Las que NO, aunque su nombre suene a número
resto = [c for c in facturas.columns if c not in numericas]

print('pandas las trata como número :', numericas)
print()
print('el resto                     :', resto)
"""),
    md("""
`tasa_morosidad_cartera` está en la segunda lista. Es un porcentaje, debería ser un número, y
`describe()` **no la mostró y no avisó**.
"""),
    code("""
facturas['tasa_morosidad_cartera'].head(8)
"""),
    md("""
Llegó con **coma decimal**: `5,7` en vez de `5.7`. Para pandas eso es texto, y a partir de ahí la
columna deja de existir para cualquier cálculo.

Este es el hallazgo más incómodo de la clase: `isna()` devuelve cero, `describe()` no reclama, y aun
así falta una columna entera del análisis. Lo único que lo detecta es **mirar el tipo**.
"""),
    code("""
anotar(
    variable='tasa_morosidad_cartera',
    categoria='formato',
    descripcion='Llega con coma decimal y pandas la lee como texto, no como número.',
    evidencia="describe() devuelve 5 columnas; select_dtypes('number') la excluye; head() muestra '5,7'",
    severidad='alta',
    impacto='Queda fuera de todo cálculo sin que nada avise. Cualquier análisis que la involucre '
            'está incompleto y parece completo.',
)
"""),
    md("""
## Paso 2 · ¿Cómo se comporta cada una?

Ya sabemos qué es cada columna. Ahora, una por una: centro, dispersión y forma.

> **Antes de ejecutar:** el monto promedio de una factura de esta cartera, ¿está más cerca del
> promedio o de la mediana? ¿Y cuánto creen que se diferencian?
"""),
    code("""
facturas['monto_neto'].describe()
"""),
    md("""
Promedio y mediana **no coinciden**, y no por poco. Vale la pena verlo:
"""),
    code("""
import matplotlib.pyplot as plt

media, mediana = facturas['monto_neto'].mean(), facturas['monto_neto'].median()

fig, ax = plt.subplots(figsize=(9, 4))
visibles = facturas.loc[facturas['monto_neto'] < 5e6, 'monto_neto']

ax.hist(visibles, bins=40, color='#2b6ca3', edgecolor='white', linewidth=0.5)
ax.axvline(mediana, color='#2e8b57', lw=2.6, label=f'mediana = ${mediana:,.0f}'.replace(',', '.'))
ax.axvline(media, color='#b5443a', lw=2.6, ls='--', label=f'promedio = ${media:,.0f}'.replace(',', '.'))
ax.set(xlabel='monto neto de la factura, en pesos', ylabel='facturas')
ax.legend(frameon=False)

# El promedio y la mediana se calculan sobre las 612; el dibujo recorta las de arriba de
# $5.000.000 para que la forma se vea. Un gráfico que filtra tiene que decir qué filtró.
ax.set_title(f'{len(visibles)} de {len(facturas)} facturas · fuera, las 4 sobre $5.000.000',
             fontsize=10, color='#777')
plt.show()
"""),
    md("""
La mitad de las facturas está bajo los **$616.000**, pero el promedio es **$1.014.250**: lo arrastra
una cola de facturas grandes.

Quien dimensione el trabajo de cobranza con el promedio está planificando para una factura que casi
no existe. El promedio no está mal calculado: **está mal usado como resumen** de una distribución
así.
"""),
    code("""
anotar(
    variable='monto_neto',
    categoria='distribución',
    descripcion='Distribución muy asimétrica: el promedio queda 65% por encima de la mediana.',
    evidencia='promedio $1.014.250 vs mediana $616.000, con cola larga a la derecha',
    severidad='media',
    impacto='Dimensionar la carga de cobranza con el promedio sobreestima la factura típica.',
)
"""),
    md("""
---

# Parte 2 · Vamos juntos

Los pasos 3 y 4. El código está escrito: esto no es una clase de pandas. Lo que hacen ustedes es
**leer la salida, decidir si es hallazgo y redactar la línea**.
"""),
    md("""
## Paso 3 · ¿Cómo se relacionan entre sí?

De a dos, y todas a la vez.

> **Antes de ejecutar:** de las cinco columnas numéricas, ¿cuáles esperan que se muevan juntas?
"""),
    code("""
facturas.select_dtypes(include='number').corr().round(3)
"""),
    md("""
`monto_neto` y `monto_total` correlacionan **1,000**. No es una relación fuerte: es la **misma
medición dos veces**.
"""),
    code("""
# round(2): a cuatro decimales aparecen 8 razones distintas, y las 8 son las facturas
# cargadas en miles, que son parte de la práctica. Acá interesa la relación, no ese ruido.
(facturas['monto_total'] / facturas['monto_neto']).round(2).value_counts()
"""),
    md("""
La razón es siempre **1,19**. Es el **IVA**. Ninguna de las dos aporta información que la otra no
tenga.

Este hallazgo ya lo vimos en clase: acá el paso 3 sirve para **registrarlo**, no para descubrirlo. Los
tres hallazgos que les tocan a ustedes vienen en el paso 4.
"""),
    code("""
# Ya resuelto en clase; queda anotado. Descomenten si quieren completarlo
# anotar(
#     variable='monto_neto / monto_total',
#     categoria='...',
#     descripcion='...',   # completar
#     evidencia='...',     # completar
#     severidad='...',     # completar
#     impacto='...',       # completar
# )
"""),
    md("""
## Paso 4 · ¿Qué se contradice?

Lo que debería cumplirse siempre y no se cumple. Tres miradas, tres técnicas.

### 4.1 · Contar: filas repetidas

> **Antes de ejecutar:** el archivo declara 612 facturas. ¿Son 612 facturas **distintas**?
"""),
    code("""
repetidas = facturas[facturas.duplicated(keep=False)]

print('Copias sobrantes    :', facturas.duplicated().sum())
print('Filas involucradas  :', len(repetidas))
"""),
    md("""
> **Antes de ejecutar la celda siguiente:** esas filas repetidas, ¿de quién son? ¿Esperan que estén
> repartidas entre todos los ejecutivos?
"""),
    code("""
repetidas['ejecutivo'].value_counts()
"""),
    md("""
**Las 24 son del mismo ejecutivo.**

Acá el hallazgo cambia de naturaleza. Doce filas repetidas es un error de datos; doce filas
repetidas **todas del mismo origen** es un hallazgo de **proceso**, y quien tiene que enterarse no
es el área de sistemas.
"""),
    code("""
anotar(
    variable='(fila completa)',
    categoria='cantidad',
    descripcion='12 facturas duplicadas exactas, y las 24 filas involucradas son del mismo ejecutivo.',
    evidencia="duplicated().sum() = 12; value_counts() del ejecutivo devuelve un solo nombre",
    severidad='alta',
    impacto='Infla el monto de la cartera y la carga aparente de ese ejecutivo. Apunta a un problema '
            'de carga de datos, no solo a filas sobrantes.',
)
"""),
    md("""
### 4.2 · Agrupar y contar valores distintos: un cliente, dos nombres

Un RUT identifica a una empresa. Entonces cada RUT debería tener **una** razón social.

> **Antes de ejecutar:** ¿cuántos RUT creen que aparecen con más de un nombre?
"""),
    code("""
nombres_por_rut = facturas.groupby('rut_cliente')['razon_social'].nunique()
inconsistentes = nombres_por_rut[nombres_por_rut > 1]

print('RUT con más de una razón social:', len(inconsistentes))
inconsistentes
"""),
    code("""
# Cómo se ven los nombres de uno de ellos
rut = inconsistentes.index[0]
facturas.loc[facturas['rut_cliente'] == rut, ['rut_cliente', 'razon_social']].drop_duplicates()
"""),
    md("""
> **Su turno:** ¿por qué esto importa para la decisión que se va a apoyar? Piensen qué pasa si
> alguien agrupa por `razon_social` en vez de por `rut_cliente`.
"""),
    code("""
# Descomenten y completen
# anotar(
#     variable='razon_social',
#     categoria='...',
#     descripcion='...',   # completar
#     evidencia='...',     # completar
#     severidad='...',     # completar
#     impacto='...',       # completar
# )
"""),
    md("""
### 4.3 · Comparar dos columnas: lo que es imposible

Una factura no puede pagarse antes de emitirse. Eso no es una regla de negocio discutible: es una
imposibilidad lógica.
"""),
    code("""
emision = pd.to_datetime(facturas['fecha_emision'])
pago = pd.to_datetime(facturas['fecha_pago'], errors='coerce')

imposibles = facturas[pago < emision]

print('Facturas pagadas antes de emitirse:', len(imposibles))
imposibles[['id_factura', 'fecha_emision', 'fecha_pago', 'dias_atraso']]
"""),
    md("""
Cinco facturas. Ninguna herramienta las marcó: las dos fechas son fechas válidas, y **la
contradicción solo aparece al compararlas entre sí**.

> **Su turno:** anoten este hallazgo. Presten atención al campo de severidad — acá hay una
> inconsistencia lógica **verificada**, no una sospecha.
"""),
    code("""
# Descomenten y completen
# anotar(
#     variable='fecha_pago / fecha_emision',
#     categoria='...',
#     descripcion='...',   # completar
#     evidencia='...',     # completar
#     severidad='...',     # completar
#     impacto='...',       # completar
# )
"""),
    md("""
---

# Parte 3 · Ahora ustedes

El paso 5, por cuenta propia.
"""),
    md("""
## Paso 5 · ¿De qué momento es cada dato?

**Fuga de datos** (*data leakage*): información en las variables predictoras que **no va a estar
disponible en el momento de decidir**, o que ya contiene la respuesta.

En este archivo hay tres sospechosas, una por cada señal. Las tres celdas de abajo revelan una cada
una.

**Obligatorio: elijan UNA**, ejecútenla, lean la salida y escriban su línea de bitácora. Lo que se
evalúa no es el código —está escrito— sino el argumento: **por qué** esa columna es sospechosa y
**qué le preguntarían al dueño del dato**.
"""),
    md("""
**Señal A · Es la respuesta con otras palabras.** Separa el objetivo sin equivocarse nunca.
"""),
    code("""
pd.crosstab(facturas['estado_cobranza'], facturas['dias_atraso'] > 0)
"""),
    md("""
**Señal B · Solo existe si el evento ocurrió.** Vale cero en todos los demás casos.
"""),
    code("""
facturas.groupby(facturas['dias_atraso'] > 0)['monto_gestion_cobranza'].describe()
"""),
    md("""
**Señal C · Es una cifra del período.** No varía dentro del mes.
"""),
    code("""
mes = facturas['fecha_emision'].str[:7]
facturas.groupby(mes)['tasa_morosidad_cartera'].nunique()
"""),
    md("""
> **Ojo con la versión fácil.** «La variable con fuga está vacía hasta que ocurre el evento» no
> sirve: la tasa de morosidad está en **todas** las filas y es sospechosa igual. Lo que llega tarde
> no es el dato: es su **valor**.

Y la respuesta **no está en el archivo**: la tiene quien es dueño del dato. De acá se sale con una
**lista de sospechosos**, no con un dictamen.
"""),
    code("""
# Su hallazgo obligatorio: una señal de fuga, sustentada · descomenten y completen
# anotar(
#     variable='...',
#     categoria='...',
#     descripcion='...',
#     evidencia='...',
#     severidad='...',
#     impacto='...',
# )
"""),
    md("""
## Un hallazgo a elección

Elijan **uno** de los tres y agréguenlo a la bitácora. Las celdas de abajo son puntos de partida:
tendrán que completarlas.

1. Los **8 montos** que están cargados en miles y no en pesos.
2. El **contrato de $148 millones**: ¿error o caso real? Argumenten.
3. La columna que **ningún estadístico está mirando**, y qué se pierde por eso.

> Sobre la tercera: la celda convierte la columna para **dimensionar lo que se está perdiendo**, no
> para arreglar el archivo. Corregirlo de verdad es la clase 4.
"""),
    code("""
# Opción 1 · los montos en otra unidad
facturas.nsmallest(12, 'monto_neto')[['id_factura', 'razon_social', 'segmento', 'monto_neto']]
"""),
    code("""
# Opción 2 · la factura más grande de la cartera
facturas.nlargest(3, 'monto_neto')[['id_factura', 'razon_social', 'segmento', 'monto_neto', 'dias_credito']]
"""),
    code("""
# Opción 3 · qué pasaría si esa columna SÍ fuera número
convertida = pd.to_numeric(facturas['tasa_morosidad_cartera'].str.replace(',', '.'), errors='coerce')
convertida.describe()
"""),
    code("""
# Su hallazgo a elección · descomenten y completen
# anotar(
#     variable='...',
#     categoria='...',
#     descripcion='...',
#     evidencia='...',
#     severidad='...',
#     impacto='...',
# )
"""),
    md("""
---

## La bitácora

Lo que se entrega. No es un archivo corregido: es la lista de lo que se encontró, con qué lo
demuestra y a qué decisión afecta.
"""),
    code("""
bitacora = pd.DataFrame(BITACORA)

print(f'{len(bitacora)} hallazgos registrados')
bitacora
"""),
    md("""
## Para cerrar

La rutina, en orden, y lo que cazó cada paso en este archivo:

| Paso | Qué se preguntó | Qué apareció |
|:--|:--|:--|
| 1 | ¿Qué es cada columna? | Un porcentaje que llegó como texto |
| 2 | ¿Cómo se comporta cada una? | Un promedio que describe una factura que casi no existe |
| 3 | ¿Cómo se relacionan? | Dos columnas que son la misma con IVA |
| 4 | ¿Qué se contradice? | Duplicados de un solo ejecutivo, RUT con dos nombres, pagos imposibles |
| 5 | ¿De qué momento es cada dato? | Tres columnas cuyo valor llega después de decidir |

El archivo declaraba **cero celdas vacías**. Ninguno de estos hallazgos aparece en `info()`.

**Nada de esto se corrigió hoy.** Decidir qué hacer con cada hallazgo —imputar, eliminar,
transformar, o preguntarle a quien es dueño del dato— es la clase 4. Lo que se lleva de acá es la
bitácora, y la rutina de cinco pasos, que sirve para cualquier archivo que les pongan por delante.
"""),
]


def main() -> None:
    if not SVG.exists():
        raise SystemExit(f"falta el diagrama {SVG}")
    nb = {
        "cells": con_ids(CELDAS),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_md = sum(1 for c in CELDAS if c["cell_type"] == "markdown")
    print(f"{SALIDA.relative_to(REPO_ROOT)}: {len(CELDAS)} celdas "
          f"({n_md} markdown / {len(CELDAS) - n_md} código)")


if __name__ == "__main__":
    main()
