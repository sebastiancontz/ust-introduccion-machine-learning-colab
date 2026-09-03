#!/usr/bin/env python3
"""Construye notebooks/02-formulacion.ipynb.

POR QUÉ EXISTE UN GENERADOR: en la clase 1 el notebook se armó con un script que quedó en un
scratchpad temporal, y al regenerarlo se revirtieron en silencio dos correcciones que solo estaban
parchadas en el .ipynb. Regla del repo: el generador se versiona JUNTO al artefacto y el .ipynb no
se edita a mano. Si hay que corregir algo, se corrige acá y se vuelve a ejecutar.

CIFRAS (regla de la clase 4, aplicada acá el 2026-08-11): ninguna magnitud se escribe a mano dentro
del código. La tabla comparativa del final rotulaba «1.221 registros» y «366 registros» como texto fijo, y
el filtro de episodios anunciaba «los 3 registros sin mes registrado» en un comentario: las tres cifras
ahora se derivan y se imprimen. Una cifra escrita a mano deriva en silencio cuando una decisión
anterior mueve la base, y entonces el notebook afirma algo que su propia salida desmiente.

FASE 5E (2026-08-11, primera revisión de consistencia derivada que pasa esta clase):
  · La grilla se arma con 33 de los 36 empleados, y por qué se cayeron los otros tres no estaba
    escrito en ninguna parte: dos no tienen ningún registro fechado y el tercero, E-08, desaparece
    recién al filtrar los registros disciplinarios, porque sus dos únicas registros son eso. Es una
    decisión de población —el mismo problema que la fecha de ingreso— y el notebook la aplicaba en
    silencio con un `!=` mientras enseñaba, en la celda siguiente, que un período sin ausencias es
    una observación válida y no un faltante. Ahora la celda imprime cuántos empleados quedan y el
    Markdown declara la decisión con su razón. NO se cambió a quién incluye la grilla: hacerlo
    mueve el 4,20 a 4,07 y el 6,2 a 6,0, y esas cifras están en las slides, en la guía y DIBUJADAS
    dentro de `unidad-observacion.svg`. Queda como decisión del docente en `TODO.md`.
  · El cierre fijaba la vara en el error del promedio, 6,2 h, cuatro celdas después de mostrar que
    la mediana se equivoca en 4,2 y de concluir que «el piso más tonto le gana al que parecía
    razonable». Un modelo entre 4,2 y 6,2 quedaba justificado por esa frase y una constante lo
    supera. La vara ahora es el mejor piso medido, como ya decían las slides y la guía.
  · El histograma recorta la cola con `clip(upper=60)` sin decirlo: 13 de los 366 registros observados
    están sobre 60 h y el máximo real es 128. El título de cada panel declara el `n` y el recorte.

COMPATIBILIDAD: nada del código depende del nombre literal de un dtype. Las listas de valores
únicos se imprimen con `tolist()`, porque la envoltura con que pandas muestra un arreglo cambia
entre versiones y Colab no siempre trae la misma que el entorno local.

Uso:  python ediciones/2026/notebooks/_build_02_formulacion.py
      (después: ejecutar el notebook para dejarlo con salidas, ver el README del repo)
"""
from __future__ import annotations

import ast
import base64
import json
import statistics
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "02-formulacion.ipynb"
SVG = REPO_ROOT / "assets" / "ilustraciones" / "unidad-observacion.svg"


def figura_base64(path: Path, alt: str, width: int = 900) -> str:
    """Diagrama embebido como data URI: el notebook queda autocontenido y no depende de Pages.

    `alt` es obligatorio: sin él la figura no existe para quien usa lector de pantalla, y
    `nbconvert` lo reclama al exportar.
    """
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<p align="center"><img src="data:image/svg+xml;base64,{b64}" '
            f'width="{width}" alt="{alt}"></p>')


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}


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
<p>Semana 02: Formulación de problemas predictivos</p>
</div>
</div>
"""),
    md("""
# 02 · Formulación de problemas predictivos

Hoy no se entrena ningún modelo. Ni una línea.

Este notebook hace tres cosas, y las tres ocurren **antes** de que exista un modelo:

1. Contar, para descubrir qué representa realmente un registro del archivo.
2. **Construir** el registro que la decisión necesita, que no es la que vino.
3. Calcular el **baseline** sobre las dos versiones y comparar.

Al final van a ver el número con que abrió la clase: dos equipos, el mismo archivo, y una diferencia
de más del triple producida por una sola decisión.
"""),
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas, así que la celda siguiente termina en segundos. Está
igual porque el notebook también tiene que correr fuera de Colab, y porque dejar escrito de qué
depende un análisis es parte de que sea reproducible.
"""),
    code("""
%%capture
!pip install -q pandas numpy matplotlib seaborn scikit-learn
"""),
    md("""
## Cargar datos

El archivo es un registro de ausencias de una empresa de servicios: tres años de historia.

Un detalle del código: `fillna` reemplaza las celdas vacías por el valor que se le pase. Acá
deja en blanco los períodos que no se pudieron reconstruir.
"""),
    code("""
import os
import pandas as pd

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO

# periodo se lee como texto: es un año-mes, no un número con el que se opere
df = pd.read_csv(BASE + 'absentismo_laboral.csv', dtype={'periodo': str})
df['periodo'] = df['periodo'].fillna('')

print('Registros y columnas:', df.shape)
df.head()
"""),
    md("""
---

# Parte 1 · ¿Qué representa un registro?

740 registros. La pregunta que decide todo lo demás es de qué son esos 740 registros.

> **Antes de ejecutar la celda siguiente, respondan:** ¿cuántos empleados distintos creen que hay?

`nunique` cuenta cuántos valores **distintos** tiene una columna, que no es lo mismo que
cuántos registros hay.
"""),
    code("""
print('Registros en el archivo :', len(df))
print('Empleados distintos :', df['id_empleado'].nunique())
"""),
    md("""
Treinta y seis personas en 740 registros. Entonces **un registro no es un empleado**: es un *episodio de
ausencia*, y una misma persona aparece muchas veces.

Veamos hasta qué punto.
"""),
    code("""
por_empleado = df['id_empleado'].value_counts()   # cuántos registros aporta cada persona

print('Registros por empleado')
print('  mínimo :', por_empleado.min())
print('  mediana:', int(por_empleado.median()))
print('  máximo :', por_empleado.max())
"""),
    md("""
### El caso testigo

Un empleado concreto, en un mes concreto. Miren la columna `motivo_ausencia`: cuenta una historia.
"""),
    code("""
caso = df[(df['id_empleado'] == 'E-03') & (df['periodo'] == '2010-02')]

print(f'{len(caso)} episodios, {caso["horas_ausencia"].sum()} horas en total')
caso[['id_empleado', 'periodo', 'dia_semana', 'motivo_ausencia', 'horas_ausencia']]
"""),
    md("""
Una dolencia musculoesquelética y una tanda de sesiones de kinesiología. Quince registros que, para la
decisión que queremos apoyar, son **un solo hecho**: este empleado faltó 45 horas ese mes.

### Y un registro tampoco es siempre una ausencia

Hay registros en la tabla que no son episodios de ausencia. Se detectan cruzando dos columnas.
"""),
    code("""
disciplinarias = df[df['falta_disciplinaria'] == 'si']

# tolist(): sin él pandas imprime su propia envoltura del arreglo, que cambia de una
# versión a otra y acá solo distrae de lo único que importa, que son los valores
print('Registros con falta disciplinaria:', len(disciplinarias))
print('Horas de ausencia que registran:', sorted(disciplinarias['horas_ausencia'].unique().tolist()))
print('Motivos que registran:', disciplinarias['motivo_ausencia'].unique().tolist())
"""),
    md("""
Cuarenta registros, todos con **cero horas** y **sin motivo**. Son registros disciplinarios que quedaron
guardados en una tabla de ausencias. No son el evento que el nombre del archivo promete, y hay que
sacarlas antes de calcular nada.

Esto no se detecta leyendo los nombres de las columnas. Se detecta contando y cruzando.
"""),
    md("""
---

# Parte 2 · Construir el registro que la decisión necesita

La decisión es **cuántos turnos de reemplazo cubrir el mes que viene**. Se toma por empleado y una
vez al mes, así que el registro tiene que ser el **empleado-mes**.

El archivo no la trae. Hay que construirla, y son dos pasos.
"""),
    md(figura_base64(
        SVG, width=900,
        alt="A la izquierda, la tabla que trae el archivo: 740 registros donde cada registro es un episodio "
            "de ausencia y el empleado E-03 aparece quince veces solo en febrero de 2010. A la "
            "derecha, la tabla que la decisión necesita: 1.221 registros donde cada registro es un empleado "
            "en un mes, incluyendo un mes con cero horas que no existía en el archivo original. "
            "Abajo, la comparación de promedios: 4,20 horas por empleado-mes sobre la tabla "
            "completa y 14,00 si solo se cuentan los meses que aparecen en el archivo")),
    md("""
### Paso 1 · Agrupar los episodios

<!-- contrato: accion=agrupar-por-empleado-mes -->

Un `groupby` por empleado y período. Antes de ejecutar, una advertencia: **este paso todavía tiene
un problema**, y es el punto de la clase. Mírenlo y sigan.
"""),
    code("""
fechado = df[df['periodo'] != '']          # sin mes registrado no hay empleado-mes que construir
fechado = fechado[fechado['falta_disciplinaria'] == 'no']   # y un registro disciplinario no es una ausencia

observados = (fechado
              .groupby(['id_empleado', 'periodo'])['horas_ausencia']
              .sum()
              .reset_index())

print(f'De los {len(df)} registros del archivo quedan {len(fechado)} episodios de ausencia fechados')
print(f'Y de los {df["id_empleado"].nunique()} empleados quedan {fechado["id_empleado"].nunique()}')
print('Registros empleado-mes que aparecen en el archivo:', len(observados))
observados.head()
"""),
    md("""
### Paso 2 · Los meses en que no pasó nada

<!-- contrato: accion=construir-la-grilla -->

366 registros. Pero hay 33 empleados con historial y 37 meses de ventana, o sea **1.221** combinaciones
posibles de empleado y mes.

Faltan 855. ¿Dónde están?

No están porque **no hubo ausencia** esos meses, y un sistema registra cuando *pasa* algo. Pero un
mes sin ausencias no es un dato faltante: es una observación perfectamente válida, y vale **cero
horas**.

Si no se construyen, el promedio se calcula solo sobre los meses malos.

Para construirlas se usa `from_product`, de `MultiIndex`: arma **todas** las combinaciones
posibles de empleado y período, incluidas las que el archivo no trae.
"""),
    code("""
empleados = sorted(fechado['id_empleado'].unique())
periodos = sorted(fechado['periodo'].unique())

print(f'{len(empleados)} empleados x {len(periodos)} meses = {len(empleados) * len(periodos)} registros posibles')
print(f'Ventana: de {periodos[0]} a {periodos[-1]}')

# from_product: TODAS las combinaciones, no solo las que el archivo trae
grilla = pd.MultiIndex.from_product([empleados, periodos], names=['id_empleado', 'periodo'])

completa = (observados
            .set_index(['id_empleado', 'periodo'])['horas_ausencia']
            .reindex(grilla, fill_value=0)      # el mes sin ausencias vale 0, no NaN
            .reset_index())

ceros = (completa['horas_ausencia'] == 0).sum()
print(f'\\nRegistros en la tabla completa: {len(completa)}')
print(f'De ellas, en cero: {ceros}  ({ceros / len(completa):.0%})')
completa.head()
"""),
    md("""
El **70 %** de la tabla son ceros que no estaban en el archivo. Esa es la tabla que la decisión
necesita, y hubo que construirla.

> **Chequeo rápido:** si un empleado entró a la empresa a mitad de la ventana, ¿corresponde ponerle
> cero en los meses anteriores? Piénsenlo. La respuesta es que **no**, y por eso este paso es una
> decisión de formulación y no un `reindex` automático. En este archivo no tenemos fecha de ingreso,
> así que asumimos que los 33 estuvieron los 37 meses; en un caso real, eso hay que verificarlo.

Y hay una segunda decisión del mismo tipo, escondida en el filtro de dos celdas más arriba: de los
36 empleados del archivo la grilla se quedó con 33. Tres se cayeron por no tener **ningún episodio
de ausencia fechado**, y uno de esos tres desapareció recién al sacar los registros disciplinarios.
Por el criterio que acabamos de aplicar a los meses, esos tres son 37 ceros cada uno y deberían
estar. Los dejamos fuera porque el archivo no permite distinguir «no faltó nunca» de «no estuvo
contratado», que es el mismo problema de la fecha de ingreso. **Es una decisión, no un descarte
técnico**, y por eso queda escrita acá y no en un comentario del código.
"""),
    md("""
---

# Parte 3 · El baseline

El baseline es la predicción más simple posible **sin modelo**. Para una cantidad, predecir siempre
el promedio.

En la clase pasada ya lo vieron funcionar: `DummyClassifier` sobre la cartera de morosidad daba 77 %
de exactitud y detectaba cero morosos. Es el mismo recurso, en su versión para cantidades.

> **Antes de ejecutar:** ¿cuál creen que va a ser el promedio de horas por empleado-mes?

<!-- contrato: accion=calcular-el-piso -->
"""),
    code("""
from sklearn.dummy import DummyRegressor

y = completa['horas_ausencia']
X = completa[['id_empleado']]        # el baseline ni las mira: predice lo mismo para todos

piso = DummyRegressor(strategy='mean').fit(X, y)   # strategy='mean': el piso de una cantidad es su promedio
prediccion = piso.predict(X)

print(f'El baseline predice {prediccion[0]:.2f} horas para todos los empleados, todos los meses.')
"""),
    md("""
Cuatro horas. Y ahora la pregunta que importa: **¿cuánto se equivoca?**

Lo medimos en horas, que es la unidad del problema y la que entiende quien decide. Las métricas con
nombre propio llegan en la clase 8.
"""),
    code("""
error_promedio = (y - prediccion).abs().mean()

print(f'Predice          : {prediccion[0]:.2f} horas')
print(f'Se equivoca en   : {error_promedio:.1f} horas, en promedio')
"""),
    md("""
### La comparación que abrió la clase

Repitamos exactamente lo mismo, pero **sobre los 366 registros que venían en el archivo**, sin los meses
en cero. Mismo método, misma librería, mismo código. Lo único distinto es qué se considera un registro.

> **Antes de ejecutar:** ¿el promedio va a subir o a bajar? ¿Y el error?

El resultado se arma como un `DataFrame` a partir de un diccionario, para poder compararlo de
un vistazo.
"""),
    code("""
y_obs = observados['horas_ausencia']
piso_obs = DummyRegressor(strategy='mean').fit(observados[['id_empleado']], y_obs)
pred_obs = piso_obs.predict(observados[['id_empleado']])

resumen = pd.DataFrame({
    'unidad de observación': [f'Tabla completa ({len(completa)} registros)',
                              f'Solo lo que vino ({len(observados)} registros)'],
    'el baseline predice': [f'{prediccion[0]:.2f} h', f'{pred_obs[0]:.2f} h'],
    'se equivoca en': [f'{error_promedio:.1f} h', f'{(y_obs - pred_obs).abs().mean():.1f} h'],
})
resumen
"""),
    md("""
**4,20 horas contra 14,00.** Más del triple, y no lo produjo ningún algoritmo: lo produjo decidir
qué es un registro.

Los dos equipos del comienzo de la clase tenían razón en sus cuentas. Uno estaba respondiendo otra
pregunta.
"""),
    md("""
### Por qué pasa: mírenlo

Un solo gráfico, y con un propósito acotado: **explicar el número de arriba**. Mirar distribuciones
de forma sistemática para auditar un archivo es otra cosa y es la clase 3; acá solo queremos ver de
dónde sale la diferencia entre 4,20 y 14,00.
"""),
    code("""
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='tab10')

fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True, sharey=True)
paneles = [('Tabla completa', completa['horas_ausencia']),
           ('Solo lo que vino en el archivo', observados['horas_ausencia'])]

for eje, (titulo, datos) in zip(ejes, paneles):
    sns.histplot(datos.clip(upper=60), bins=30, ax=eje, color='tab:blue')
    eje.axvline(datos.mean(), color='tab:red', linestyle='--', linewidth=2,
                label=f'promedio = {datos.mean():.2f} h')
    # un gráfico que recorta declara qué recortó, y sobre cuántos registros está calculado
    eje.set_title(f'{titulo}\\nn = {len(datos)} · las {(datos > 60).sum()} sobre 60 h se apilan a la derecha',
                  fontsize=10)
    eje.set_xlabel('horas de ausencia en el mes')
    eje.legend()

ejes[0].set_ylabel('empleado-mes')
plt.tight_layout()
plt.show()
"""),
    md("""
A la izquierda, la barra de la izquierda del todo son los 855 meses sin ausencias. Son la mayoría, y
tiran el promedio hacia abajo. A la derecha esos meses simplemente no existen, así que el promedio
se calcula solo sobre los meses en que alguien faltó.

### Un detalle que vale la pena: el promedio no es el único piso
"""),
    code("""
mediana = y.median()

print(f'Predecir siempre el PROMEDIO ({y.mean():.2f} h) -> se equivoca en {(y - y.mean()).abs().mean():.1f} h')
print(f'Predecir siempre la MEDIANA  ({mediana:.2f} h) -> se equivoca en {(y - mediana).abs().mean():.1f} h')
"""),
    md("""
La mediana es **cero** —la mayoría de los meses no falta nadie— y aun así se equivoca menos que el
promedio. El piso más tonto le gana al que parecía razonable.

No es una curiosidad estadística: es la consecuencia directa de que el 70 % de los registros valen cero.
Y deja instalado que **elegir el baseline ya es una decisión**, no un trámite.
"""),
    md("""
---

# Parte 4 · Ahora ustedes

Una sola tarea, con lo que acabamos de ver.

El **momento de la predicción** es el cierre de cada mes: ahí se decide cuántos turnos cubrir para el
mes siguiente. La pregunta es qué variables están disponibles en ese instante.

**Tarea:** revisen las columnas del archivo y armen dos listas.

```python
df.columns.tolist()
```

- Las que **sí** pueden usarse como predictoras, porque ya estaban registradas antes del cierre.
- Las que **no**, porque solo se conocen después de que la ausencia ocurrió.

Escriban las dos listas en la celda de abajo, como comentario o como listas de Python.

> **Y la pregunta que importa más que el código:** hay una columna que sería con casi total
> seguridad la más informativa de todas para predecir las horas de ausencia. ¿Cuál es, y por qué hay
> que descartarla igual?
"""),
    code("""
# escriban acá su respuesta
"""),
    md("""
---

## Para cerrar

Lo que hicimos hoy, en orden:

| | |
|:--|:--|
| Contamos | y descubrimos que un registro no era un empleado |
| Construimos | el registro que la decisión necesitaba, con sus ceros |
| Calculamos | el piso contra el que se comparará cualquier modelo |

Ninguna de las tres cosas necesitó un algoritmo, y las tres cambian el resultado de todo lo que
venga después.

Y la vara la fija el **mejor** de los dos pisos que medimos, no el primero: el promedio se equivoca
en 6,2 horas, pero la mediana se equivoca en **4,2**. En las próximas clases, cualquier modelo que
construyamos sobre este problema tiene que bajar de esas 4,2 horas para justificar su costo.
Ganarle solo al promedio no alcanza, porque ya hay una constante que lo hace.
"""),
    md("""
## Atribución de datos

- **Creador:** Martiniano, A. y Ferreira, R. (2018)
- **Fuente:** [UCI Machine Learning Repository — *Absenteeism at Work*, dataset 445](https://archive.ics.uci.edu/dataset/445/absenteeism+at+work)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** los 740 registros y 21 columnas se conservaron, traducidas al español y con las
  categóricas recodificadas de números a texto; se agregó una columna `periodo` reconstruida.
"""),
]


def llamada(fuente: str, nombre: str) -> dict:
    """Deriva funcion/argumentos/cantidad del AST. NO se escribe a mano: un contrato con la firma
    de una versión anterior pasa el chequeo describiendo otro notebook."""
    limpio = "\n".join(l for l in fuente.splitlines() if not l.lstrip().startswith(("%", "!")))
    firmas = [
        len(n.args) + len(n.keywords)
        for n in ast.walk(ast.parse(limpio))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == nombre
    ]
    assert firmas, f"no hay llamadas a {nombre}: el contrato no puede declararlas"
    distintas = set(firmas)
    assert len(distintas) == 1, (
        f"las llamadas a {nombre} tienen {sorted(distintas)} argumentos; un solo número no las describe")
    return {"funcion": nombre, "argumentos": firmas[0], "cantidad": len(firmas)}


def contrato(celdas: list[dict]) -> dict:
    """Deriva `metadata.curso_contrato` de la lista FINAL de celdas.

    Se calcula, no se escribe a mano: agregar, quitar o mover una celda lo actualiza solo.
    """
    codigo = [c for c in celdas if c["cell_type"] == "code"]
    fuente = "\n".join("".join(c["source"]) for c in codigo)
    lineas = [
        len([l for l in "".join(c["source"]).splitlines()
             if l.strip() and not l.strip().startswith("#")])
        for c in codigo
    ]
    mediana = statistics.median(lineas)
    return {
        "forma": {
            "markdown": sum(1 for c in celdas if c["cell_type"] == "markdown"),
            "codigo": len(codigo),
            "llamadas": [llamada(fuente, "DummyRegressor")],
        },
        "acciones": [
            {"marcador": "agrupar-por-empleado-mes", "codigo": "groupby"},
            {"marcador": "construir-la-grilla", "codigo": "reindex"},
            {"marcador": "calcular-el-piso", "codigo": "DummyRegressor"},
        ],
        # Es una MEDIANA, no un máximo por celda: deja pasar el setup largo y sigue exigiendo que el
        # cuaderno esté partido en unidades chicas. Se fija al valor real medido, con un margen de 1.
        "max_mediana_lineas_codigo": int(mediana) + 1,
    }


def main() -> None:
    celdas = con_ids(CELDAS)
    nb = {
        "cells": celdas,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "curso_contrato": contrato(celdas),
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
