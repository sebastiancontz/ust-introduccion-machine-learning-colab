#!/usr/bin/env python3
"""Construye notebooks/04-preparacion.ipynb.

POR QUÉ EXISTE UN GENERADOR: el `.ipynb` no se edita a mano. Si hay que corregir algo, se corrige
acá y se vuelve a ejecutar. En la clase 1 se perdieron dos correcciones por parchear el artefacto.

ESTRUCTURA: las siete decisiones de la `practica` del temario, en sus tres tiempos de liberación
gradual —dos las muestra el docente, tres se hacen en conjunto, dos las toman por su cuenta—. El
resultado es DOBLE: el archivo corregido y la bitácora que lo justifica.

NO ES UN INSTRUMENTO DE EVALUACIÓN (temario, corregido el 2026-08-06): la Unidad 1 se evalúa solo
con la Solemne 1. El material del estudiante no dice "entregable", "se evalúa" ni "se entrega".

PUERTA DE EXPORTACIÓN (2026-08-09): el cuaderno promete siete decisiones y, ejecutado sin la
intervención del estudiante, solo registra cinco. Antes escribía igual el Parquet y la bitácora, o
sea presentaba como completo un resultado parcial. Ahora `DECISIONES_ESPERADAS` cierra la salida:
con menos de siete no se escribe nada y se imprime qué falta.

RECUPERACIÓN DE `descripcion_producto` (2026-08-09): NO se puede rellenar por `codigo_producto` con
  `.first()`. Medido sobre el archivo: 650 de los 4.070 códigos tienen más de una descripción, y de
  los 1.454 registros vacíos, 309 tienen un código que conoce varios nombres — incluidos casos como el 10080,
que conoce `GROOVY CACTUS INFLATABLE` y `check`. `.first()` habría rellenado la anotación del bodeguero
como si fuera el nombre del producto, contaminando con la Decisión 4 justo lo que la Decisión 1
arregla. Se automatiza SOLO lo inequívoco (código con un único nombre conocido): 1.033 registros. Los
309 ambiguas y las 112 sin catálogo quedan explícitas como pendientes de revisión.

FRONTERAS, heredadas del temario y verificadas al escribir:
  · No se escala, no se codifica y no se crean variables: eso es la clase 5.
  · No se parte en entrenamiento y prueba, no se nombran mecanismos de fuga y no se mide ningún
    modelo: eso es la clase 6. La regla operativa se enuncia y se aplica; el porqué es de allá.
  · Los casos de la TEORÍA no se repiten acá: el par de 80.995, el ajuste de −9.600 y el cruce
    23343/20713 ya se decidieron en las slides. Acá van las siete que quedaron.

CIFRAS: las que el texto afirma están recalculadas contra `datasets/ventas_online.parquet` el
2026-08-09. Las que dependen del estado de `limpio` NO se escriben en el texto: las imprime la celda,
porque cada decisión mueve la base de la siguiente. Si el dataset se regenera hay que volver a
verificar las que sí están escritas.

COMPATIBILIDAD: nada depende del nombre literal de un dtype (pandas 3 informa `str` donde pandas 2
informa `object`). El archivo es Parquet, así que los tipos vienen guardados y no dependen de la
lectura.

Uso:  python ediciones/2026/notebooks/_build_04_preparacion.py
      (después: ejecutar el notebook para dejarlo con salidas)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO_ROOT = AQUI.parents[2]
SALIDA = AQUI / "04-preparacion.ipynb"
SVG = REPO_ROOT / "assets" / "ilustraciones" / "c04-arbol-decision.svg"


def figura_base64(path: Path, alt: str, width: int = 900) -> str:
    """Diagrama embebido como data URI: el notebook queda autocontenido y no depende de Pages.

    `alt` es obligatorio: sin él la figura no existe para quien usa lector de pantalla.
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
    """nbformat 4.5+ exige un id por celda. Deterministas: regenerar no ensucia el diff."""
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
<p>Semana 04: Preparación de datos</p>
</div>
</div>
"""),
    md("""
# 04 · Preparación de datos

La clase pasada **detectamos y documentamos**. Hoy **decidimos y corregimos**.

El archivo es un año de facturación de un mayorista de artículos de regalo: **541.910 registros** en
25.900 facturas. Cada registro es **una línea de factura** —un producto dentro de una factura—, así que
una misma factura ocupa varios registros. No es una venta completa, y no es un cliente.

## Qué queda al terminar

Dos cosas, y una sin la otra no sirve:

1. el **archivo corregido**;
2. la **bitácora** que justifica cada decisión.

El archivo solo dice *qué* quedó distinto. La bitácora dice **por qué**, y es lo único que permite
que otra persona —o ustedes mismos en tres meses— entiendan las decisiones. Lo difícil de hoy no es
la técnica: es sostener el **porqué** de cada una.

## Cómo se trabaja

Siete decisiones en tres tiempos: **dos** las toma el docente en pantalla, **tres** se resuelven
entre todos, y **dos** quedan para ustedes.

Las siete cuentan. La celda final del cuaderno **no exporta nada mientras falte alguna**: un archivo
corregido a medias, sin las decisiones que lo explican, es exactamente lo que esta clase enseña a no
producir.

> **Datos reales**, no un ejemplo armado para practicar. La atribución completa está al final del
> cuaderno.
"""),
    md("""
## El árbol de decisión

Las tres preguntas de la clase, que son lo único de hoy que sirve para un archivo que no es este.
"""),
    md(figura_base64(
        SVG, width=980,
        alt="Árbol de decisión de limpieza. Tres preguntas en cadena: uno, cómo se generó este dato, "
            "quién lo escribió, cuándo y con qué sistema; dos, si estará el dato el día de la "
            "decisión o se llena recién después del hecho; y tres, qué efecto tiene cada opción. De "
            "la segunda pregunta sale una rama roja rotulada no, hacia una caja aparte: se excluye y "
            "se documenta, no se discute estrategia. De la tercera cuelgan tres opciones: excluir, "
            "que pierde filas y segmentos; imputar, cuyas versiones simples aplanan la variabilidad; "
            "y marcar, que conserva la señal del vacío y suma una columna. Al pie: ninguna es gratis, "
            "la decisión es cuál costo se acepta y por qué; y las tres preguntas se responden con "
            "conocimiento del proceso, no con estadística")),
    md("""
## Preparación del entorno

En Colab estas librerías ya vienen instaladas, así que la celda termina en segundos. Está igual
porque dejar escrito de qué depende un análisis es parte de que sea reproducible.
"""),
    code("""
%%capture
!pip install -q pandas pyarrow
"""),
    md("""
## Cargar datos

El archivo viene en **Parquet**, que guarda el tipo de cada columna. Eso elimina de entrada toda una
familia de problemas —adivinar separadores, formatos de fecha, qué texto representa un vacío— y deja
a la vista la familia que importa hoy: la del **dato mismo**. Se lee con **`read_parquet`**, que es a
Parquet lo que `read_csv` es a un CSV.

Un gesto que va a repetirse toda la clase: una **condición** no devuelve un dato, devuelve una
**máscara** —True o False por fila—, y usarla entre corchetes deja solo lo que cumple.
"""),
    code("""
import os
import pandas as pd

REPO = 'https://raw.githubusercontent.com/sebastiancontz/ust-introduccion-machine-learning-colab/main/ediciones/2026/datasets/'
BASE = '../datasets/' if os.path.exists('../datasets') else REPO
"""),
    code("""
ventas = pd.read_parquet(BASE + 'ventas_online.parquet')   # como read_csv, pero trae los tipos

ventas.head()
"""),
    md("""
`info()` es la primera lectura del archivo: cuántos registros hay, qué tipo tiene cada columna y cuántos
valores no vacíos trae. Las ocho columnas de acá abajo son las mismas que están en las slides.
"""),
    code("""
ventas.info()
"""),
    code("""
print('Registros:', len(ventas), '· Facturas:', ventas['n_factura'].nunique())
"""),
    code("""
vacias = ventas.isna().sum()

print('Celdas vacías por columna:')
print(vacias[vacias > 0])   # la máscara deja solo las columnas que sí tienen vacíos
"""),
    md("""
Seis de las ocho columnas no tienen una sola celda vacía. Los vacíos están en dos, y una está
**dentro** de la otra: eso se vio en clase y se decide más abajo.
"""),
    md("""
## La bitácora de limpieza

Es la bitácora de la clase 3 con tres campos nuevos. Allá se documentó **qué se observó**; acá,
**qué se decidió hacer**.

La función pide cinco campos, y ninguno es burocracia: sin `proceso generador` no se puede
justificar nada, y sin `efecto esperado` no se puede auditar la decisión después.

**«Se imputó con la mediana» no es una justificación.** Es la técnica. La justificación dice por qué
esa decisión y no otra, en términos del negocio.

Una convención del cuaderno, para que no sorprenda: **cada decisión ocupa dos celdas**, una que
limpia y otra que anota. Son dos actos distintos, y separarlos tiene una ventaja práctica — volver a
ejecutar la limpieza las veces que haga falta no duplica la línea de bitácora.
"""),
    code("""
DECISIONES_ESPERADAS = 7   # las siete de la clase; la celda final revisa que estén todas

BITACORA = []

"""),
    code("""
def anotar(variable, proceso_generador, decision, justificacion, efecto_esperado):
    \"\"\"Agrega una decisión a la bitácora y la devuelve para revisarla.

    variable          : la columna afectada
    proceso_generador : por qué ocurre, según el negocio. Si no se sabe, decirlo así
    decision          : qué se hizo
    justificacion     : por qué esa y no otra. La técnica NO es una justificación
    efecto_esperado   : qué cambia en los datos y en quién queda representado
    \"\"\"
    BITACORA.append({'variable': variable,
                     'proceso generador': proceso_generador, 'decisión': decision,
                     'justificación': justificacion, 'efecto esperado': efecto_esperado})
    return pd.DataFrame(BITACORA[-1:])
"""),
    md("""
Y una copia del archivo para trabajar, con **`copy()`**. **El original no se toca**: si una decisión
sale mal, hay que poder volver.
"""),
    code("""
limpio = ventas.copy()
print('Copia de trabajo:', limpio.shape)
"""),
    md("""
---

# Parte 1 · Miren esto conmigo

Dos decisiones con el ciclo completo: qué se encontró, por qué ocurre, qué se decide, por qué esa y
no otra, y qué cambia.
"""),
    md("""
## Decisión 1 · Las descripciones vacías

`descripcion_producto` tiene celdas vacías. Antes de decidir, hay que ver **de qué tamaño** es el
problema y **dónde** está.
"""),
    code("""
sin_desc = limpio['descripcion_producto'].isna()

# .sum() sobre una máscara cuenta los True, porque False vale 0 y True vale 1
print('Registros sin descripción:', sin_desc.sum())
print(f'Porcentaje del archivo: {100 * sin_desc.mean():.2f} %')
"""),
    code("""
print('De esos registros:')

# .str agrupa las operaciones de texto de una columna: acá, si empieza con 'C'
pd.DataFrame({
    'registros': [
        (sin_desc & limpio['id_cliente'].isna()).sum(),
        (sin_desc & (limpio['precio_unitario'] == 0)).sum(),
        (sin_desc & limpio['n_factura'].str.startswith('C')).sum(),
    ],
}, index=['sin cliente', 'con precio cero', 'con prefijo C'])   # index= nombra las filas
"""),
    md("""
Dos cifras coinciden con el total y la tercera es cero: **los 1.454 registros sin descripción no tienen
cliente, tienen precio cero, y ninguna lleva prefijo de cancelación**. Una firma que se repite así en tres columnas no
es azar; es compatible con un proceso común, aunque confirmarlo le toca a quien conoce el dato.

Ahora, ¿se pueden recuperar? El `codigo_producto` de esos registros aparece en otros que sí traen la
descripción, así que el código podría dar el nombre.

**Pero antes de rellenar hay que preguntarse si el código responde una sola cosa.** La celda siguiente
cuenta, para cada código, cuántos nombres distintos se le conocen, y usa **`map()`** para traducir
cada código por ese recuento.
"""),
    code("""
# cuántos nombres DISTINTOS conoce cada código, contando solo los registros que sí traen descripción
nombres_por_codigo = (limpio.loc[~sin_desc]
                      .groupby('codigo_producto')['descripcion_producto'].nunique())

# para cada registro vacío: cuántos nombres conoce su código (vacío = el código no está en ninguno)
conocidos = limpio.loc[sin_desc, 'codigo_producto'].map(nombres_por_codigo)

"""),
    code("""
pd.DataFrame({
    'registros': [
        (conocidos == 1).sum(),
        (conocidos > 1).sum(),
        conocidos.isna().sum(),
    ],
}, index=['código con UN nombre conocido', 'código con VARIOS nombres', 'código sin catálogo'])
"""),
    md("""
Solo el primer grupo se puede completar sin inventar nada. Los otros dos, no — y conviene ver por qué.
"""),
    code("""
# los códigos ambiguos que aparecen entre los registros vacíos, con todos sus nombres conocidos
ambiguos = limpio.loc[sin_desc & (conocidos > 1), 'codigo_producto'].unique()

print('Códigos ambiguos entre los registros a completar:', len(ambiguos))
"""),
    code("""
for cod in sorted(ambiguos)[:3]:
    nombres = sorted(limpio.loc[~sin_desc & (limpio['codigo_producto'] == cod),
                                'descripcion_producto'].unique())
    print(f'  {cod}: {nombres}')
"""),
    md("""
Ahí está el problema, y la salida lo muestra. Al código **10080** se le conocen dos nombres:
`GROOVY CACTUS INFLATABLE` y `check`. Uno es el producto; el otro es una **anotación del bodeguero**
escrita en el campo equivocado — el mismo defecto que se decide más abajo, en la Decisión 4.

Tomar «el primero que aparezca» habría rellenado 309 registros con lo que hubiera quedado antes en el
archivo, que en varios casos es la anotación y no el producto. **El orden de las filas no es un
criterio de negocio.**

Así que se completa **solo lo inequívoco**: los códigos con un único nombre conocido. El resto queda
**sin completar**, y eso es una respuesta honesta y no un vacío olvidado: su código no da un nombre
único, así que rellenarlo exigiría una decisión que no se puede tomar desde el archivo.

Esto **no es imputación estadística**. No se rellena con la moda ni con un promedio: se completa con
una **regla de negocio** —el código manda, cuando el código dice una sola cosa—, que es información
real y no una estimación.
"""),
    code("""
# el catálogo se arma SOLO con los códigos que tienen un único nombre conocido
codigos_inequivocos = nombres_por_codigo[nombres_por_codigo == 1].index
catalogo = (limpio.loc[~sin_desc & limpio['codigo_producto'].isin(codigos_inequivocos)]
            .groupby('codigo_producto')['descripcion_producto'].first())

"""),
    code("""
# map() traduce cada código por su nombre del catálogo; fillna() lo usa SOLO donde faltaba
limpio['descripcion_producto'] = limpio['descripcion_producto'].fillna(
    limpio['codigo_producto'].map(catalogo)
)

"""),
    code("""
completadas = sin_desc.sum() - limpio['descripcion_producto'].isna().sum()

print('Registros completados desde el catálogo:', completadas)
print('Registros que siguen sin descripción :', limpio['descripcion_producto'].isna().sum())
print('  (de código ambiguo o sin catálogo: requieren revisión o una decisión documentada)')
"""),
    code("""
anotar(
    variable='descripcion_producto',
    proceso_generador='registros que no son ventas y se cargaron sin nombre de producto',
    decision=f'completar SOLO desde códigos con un único nombre conocido: {completadas} registros. Los códigos con varios nombres no se rellenan',
    justificacion='cuando el código conoce un solo nombre, ese nombre es información real; cuando conoce varios, elegir por orden de aparición no es un criterio de negocio y arriesga cargar una anotación del bodeguero como si fuera el producto',
    efecto_esperado=f'ninguna fila se pierde; quedan {limpio["descripcion_producto"].isna().sum()} sin descripción, sin completar a ciegas: su código no da un nombre único y la decisión queda pendiente',
)
"""),
    md("""
## Decisión 2 · Una columna que codifica dos hechos

`n_factura` guarda el número del documento **y**, con el prefijo `C`, si la venta fue cancelada. Dos
hechos en una columna no se pueden contar por separado.
"""),
    code("""
es_cancelacion = limpio['n_factura'].str.startswith('C')

print('Líneas con prefijo C:', es_cancelacion.sum())
print('De esas, con cantidad positiva:', (es_cancelacion & (limpio['cantidad'] > 0)).sum())
"""),
    md("""
Cero excepciones: **toda línea cancelada tiene cantidad negativa**. La regla se cumple en ese
sentido, y conviene anotarlo — una auditoría también documenta lo que **sí** está bien.

Separar los dos hechos en dos columnas no pierde nada y habilita contar cada uno.
"""),
    code("""
limpio['cancelada'] = es_cancelacion

# removeprefix('C') quita UNA 'C' inicial. No usar lstrip('C'): recibe un conjunto de caracteres,
# no un prefijo, y con 'CC12345' devolvería '12345' en vez de 'C12345'
limpio['n_documento'] = limpio['n_factura'].str.removeprefix('C')

print(limpio[['n_factura', 'n_documento', 'cancelada']].head(3).to_string(index=False))
"""),
    md("""
**Una advertencia sobre `cancelada`, que depende de para qué se use.**

Acá es una corrección de formato: separa dos hechos que venían pegados en una columna. Pero si el
problema fuera **anticipar qué pedidos se van a cancelar**, esta columna —y la cantidad negativa que
la acompaña— **serían la respuesta escrita de otra forma**: se llenan cuando la cancelación ya
ocurrió, no antes.

La pregunta de la clase 3 lo detecta: **¿con información de qué momento se llena este campo?** Una
columna que solo existe después del hecho no puede usarse para predecirlo. Separarla está bien;
tratarla como predictora sin hacerse esa pregunta, no.
"""),
    code("""
anotar(
    variable='n_factura',
    proceso_generador='el sistema marca la cancelación con un prefijo en vez de una columna propia',
    decision='separar en n_documento y cancelada; se conserva n_factura',
    justificacion='dos hechos en una columna no se pueden contar ni filtrar por separado',
    efecto_esperado='se puede contar cancelaciones sin parsear texto; no se pierde información. Queda registrado que cancelada es posterior al hecho y no sirve para anticiparlo',
)
"""),
    md("""
---

# Parte 2 · Lo hacemos juntos

Tres decisiones, una por tipo de problema. El código está escrito: lo que hacen ustedes es **decidir
antes de ejecutar**, y después leer la salida.
"""),
    md("""
## Decisión 3 · Los duplicados

**Antes de ejecutar, decidan.** El archivo tiene miles de filas idénticas en todas sus columnas.
¿Se borran las repeticiones, se conservan, o falta información para responder? Comprométanse con una
respuesta antes de mirar la salida.

Y una precisión sobre qué se está contando, porque tres llamadas casi iguales dan tres números:

- **`duplicated()`** marca cada fila que **ya apareció antes**, así que no marca la primera de cada
  grupo. Cuenta las **repeticiones**, que es lo que se eliminaría.
- **`duplicated(keep=False)`** marca **todas** las filas de cada grupo repetido, incluida la primera:
  el **tamaño del conjunto afectado**, no cuántas se van.
- **`duplicated(subset=[...])`** compara **solo esas columnas** en vez de la fila completa.
"""),
    code("""
print('Repeticiones (al conservar la primera de cada grupo):', limpio.duplicated().sum())
print('Filas involucradas en algún grupo repetido        :', limpio.duplicated(keep=False).sum())

"""),
    code("""
clave = ['n_factura', 'codigo_producto', 'cantidad', 'precio_unitario']
print()
print('Repeticiones usando solo la clave de negocio', clave, ':')
print(' ', limpio.duplicated(subset=clave).sum())
"""),
    md("""
Dos números distintos para la misma pregunta, y la diferencia es el contenido de esta decisión.

`duplicated()` compara la **fila completa**. Con `subset` compara solo la clave de negocio, y
aparecen **más repeticiones**: filas que son el mismo hecho pero difieren en alguna columna.

Y la pregunta de fondo no la resuelve pandas: **una factura puede tener legítimamente dos líneas del
mismo producto** —dos cajas cargadas por separado—. Si eso es posible en este negocio, borrar por
clave de negocio elimina ventas reales.
"""),
    code("""
antes = len(limpio)
limpio = limpio.drop_duplicates()   # sin subset: compara la fila completa, la opción conservadora

print('Filas antes :', antes)
print('Filas después:', len(limpio))
"""),
    code("""
anotar(
    variable='(todas)',
    proceso_generador='cargas repetidas del mismo archivo; no se distingue de una recompra en el mismo minuto',
    decision=f'eliminar las {antes - len(limpio)} repeticiones de FILA COMPLETA, conservando la primera de cada grupo',
    justificacion='una factura puede tener dos líneas del mismo producto, así que borrar por clave de negocio eliminaría ventas reales',
    efecto_esperado=f'el archivo baja a {len(limpio)} filas; los hechos repetidos dejan de pesar doble',
)
"""),
    md("""
**Desde acá, las cifras del cuaderno y las de las slides dejan de coincidir exactamente.** Las
slides cuentan sobre el archivo original y nosotros acabamos de eliminar repeticiones, así que todo
lo que se cuente de ahora en adelante sale un poco más bajo. No es un error: **cada decisión mueve la
base de la siguiente**, y esa es una razón más para dejarlas anotadas.

## Decisión 4 · Las anotaciones escritas a mano

`descripcion_producto` no siempre trae un nombre de producto. A veces trae una **nota del
bodeguero**: son las mismas que ensuciaron el catálogo en la Decisión 1.

**Antes de ejecutar, decidan.** ¿Se arregla pasando todo a minúsculas y quitando espacios?
"""),
    code("""
notas = ['check', 'damages', 'damaged', '?', 'Found', 'found']

# value_counts() cuenta cada valor; reindex() deja solo los de la lista, en ese orden
print(limpio['descripcion_producto'].value_counts().reindex(notas, fill_value=0))

"""),
    code("""
print('Códigos con más de una descripción distinta:',
      (limpio.groupby('codigo_producto')['descripcion_producto'].nunique() > 1).sum(),
      'de', limpio['codigo_producto'].nunique())
"""),
    md("""
Acá hay **dos problemas distintos** y conviene no confundirlos.

`Found` y `found` son **la misma palabra escrita de dos formas**: eso lo resuelve normalizar a
minúsculas. Pero `check` **no es un producto**, y ninguna normalización lo va a convertir en uno.

Normalizar el texto es barato y arregla lo primero. Lo segundo necesita una decisión.
"""),
    code("""
# strip() quita espacios de los extremos ('check ' != 'check'); lower() pasa a minúsculas
# va en una columna nueva para no perder el texto original
limpio['descripcion_norm'] = limpio['descripcion_producto'].str.strip().str.lower()

print('Descripciones distintas antes :', limpio['descripcion_producto'].nunique())
print('Descripciones distintas después:', limpio['descripcion_norm'].nunique())
"""),
    code("""
anotar(
    variable='descripcion_producto',
    proceso_generador='el campo del nombre se usó como cuaderno de notas cuando no había dónde anotar',
    decision='normalizar a minúsculas y sin espacios sobrantes en una columna nueva; las anotaciones no se borran',
    justificacion='normalizar unifica variantes de la misma palabra, pero no convierte una nota en un producto',
    efecto_esperado='menos categorías duplicadas; los registros anotados quedan identificables para decidirlos',
)
"""),
    md("""
## Decisión 5 · Los códigos que no son productos

`codigo_producto` guarda productos —cinco dígitos— y también cosas que no lo son.

**Antes de ejecutar, decidan.** Si un registro no corresponde a un producto, ¿se borra, se deja como
está, o se hace otra cosa con ella?
"""),
    code("""
# str.match ancla al INICIO del texto, así que \\d{5} pide cinco dígitos ahí mismo (no hace falta
# escribir ^). La virgulilla ~ invierte la máscara: quedan los que NO calzan
no_producto = ~limpio['codigo_producto'].str.match(r'\\d{5}')

print(limpio.loc[no_producto, 'codigo_producto'].value_counts().head(8).to_string())
print()
print('Líneas que no son producto:', no_producto.sum())
"""),
    md("""
`POST` es franqueo, `DOT` es un cargo por transporte, `M` es un ajuste manual, `BANK CHARGES` son
comisiones bancarias.

**No se borran.** Son hechos contables reales, solo que cargados en una tabla de ventas. Borrarlos
haría desaparecer costos que la empresa efectivamente tuvo. Se **separan**, que es distinto.

**Y la regla no es perfecta, que es justamente el punto.** Entre las marcadas hay códigos que **sí
son productos** —`DCGSSGIRL` es `GIRLS PARTY BAG`, `PADS` es `PADS TO MATCH ALL CUSHIONS`— y los
`gift_0001_*`, que son vales de regalo: **ingreso, no costo**. «Cinco dígitos» es una heurística, no
una definición del negocio. Marcar en vez de borrar deja esos casos recuperables; borrarlos habría
sido irreversible, y nadie se habría enterado.
"""),
    code("""
limpio['es_producto'] = ~no_producto

print(limpio['es_producto'].value_counts().to_string())
"""),
    code("""
anotar(
    variable='codigo_producto',
    proceso_generador='la tabla de ventas se usa también para cargar franqueo, transporte y ajustes contables',
    decision=f'marcar las {no_producto.sum()} con la columna es_producto; no se eliminan',
    justificacion='la mayoría son hechos contables reales y borrarlos haría desaparecer costos que la empresa tuvo; además la regla de los cinco dígitos es una heurística que deja adentro algunos productos y vales de regalo, y marcar en vez de borrar los mantiene recuperables',
    efecto_esperado='un análisis de productos puede filtrarlas; la contabilidad no pierde nada',
)
"""),
    md("""
---

# Parte 3 · Ahora ustedes

Dos decisiones, y las dos van a la bitácora. **La primera es obligatoria.**
"""),
    md("""
## Antes de decidir: ¿por qué falta?

La estrategia para un vacío no se elige por el porcentaje. Se elige por **el motivo de la ausencia**.
Tres casos, con los nombres que van a encontrar en cualquier documentación:

- **Ausencia pareja** (*MCAR*, «faltante completamente al azar»): la probabilidad de que falte es la
  misma en todas las filas.
- **Ausencia que depende de lo observado** (*MAR*, «faltante al azar»): depende de algo que el archivo
  **sí** registra en otra columna.
- **Ausencia informativa** (*MNAR*, «faltante no al azar»): depende de algo que el archivo **no**
  registra.

**Qué se puede ver en el archivo y qué no**, que es la parte que importa:

- Se puede **descartar el primero**: si los registros con vacío se comportan distinto de los demás, la
  ausencia no es pareja. Eso lo muestra la celda de abajo, con este archivo.
- **Separar los dos últimos no se mira, se pregunta.** Depende de algo que el archivo no contiene, así
  que ningún cálculo lo decide: la respuesta la tiene quien conoce el proceso que generó el dato.

*(Los tres nombres son vocabulario estándar fuera de la bibliografía del curso; se usan acá porque
van a encontrarlos escritos así.)*
"""),
    md("""
## Decisión 6 · Obligatoria — el `id_cliente` que falta

Es la decisión más difícil de la clase, porque es la única que obliga a pronunciarse sobre **a quién
se deja fuera**.

**`id_cliente` es un identificador, no un predictor.** No mide nada del cliente: lo nombra. Esa
distinción decide la estrategia, y tiene una consecuencia inmediata — **imputar no está entre las
opciones**. Rellenar un identificador con el valor más frecuente inventa un cliente que no existe y
funde compras de personas distintas en una sola.

Un detalle antes de mirar las cifras: en la clase vieron **24,9 %**, y acá va a salir un poco más
alto. No es un error — ya eliminamos las repeticiones, así que el denominador cambió. **Cada decisión
mueve la base de la siguiente**, y esa es una razón más para dejarlas anotadas.
"""),
    code("""
sin_cliente = limpio['id_cliente'].isna()

grupo_con = limpio[~sin_cliente]   # ~ invierte la máscara: las que SÍ traen cliente
grupo_sin = limpio[sin_cliente]

print(f'Registros sin cliente: {sin_cliente.sum()} ({100 * sin_cliente.mean():.1f} %)')

"""),
    code("""
pd.DataFrame({
    'unidades por línea': [
        grupo_con['cantidad'].mean(),
        grupo_sin['cantidad'].mean(),
    ],
    'monto de la línea (GBP)': [
        (grupo_con['cantidad'] * grupo_con['precio_unitario']).mean(),
        (grupo_sin['cantidad'] * grupo_sin['precio_unitario']).mean(),
    ],
    'líneas con precio cero': [
        (grupo_con['precio_unitario'] == 0).sum(),
        (grupo_sin['precio_unitario'] == 0).sum(),
    ],
}, index=['con cliente', 'sin cliente']).round(2)
"""),
    md("""
Los dos grupos **no se comportan igual**. Con eso queda descartada la ausencia pareja: lo que falta
no falta al azar.

Las tres opciones defendibles, y ninguna es gratis:

1. **Excluir** esas filas, asumiendo por escrito a quién se deja fuera.
2. **Conservarlas y marcar** la ausencia con una columna indicadora.
3. Tratar **«sin cliente» como una categoría propia**, si resulta ser una forma de operar y no un
   error.

La celda siguiente trae **las tres escritas y listas para ejecutar**. Descomenten **una** —o escriban
la suya— y ejecútenla.

**Sobre la opción 3, un cuidado técnico:** `id_cliente` es una columna decimal (`17850.0`). Escribir
la palabra «sin cliente» dentro de ella mezclaría texto con números y rompería la columna como
identificador. La categoría va en una **columna aparte**, que es lo que hace la plantilla.

**Y lo que la justificación tiene que hacer**, además de defender la elegida: **descartar por escrito
las otras dos**, con su razón. Una decisión sin alternativas descartadas no es una decisión, es una
preferencia.
"""),
    code("""
# SU DECISIÓN · descomenten UNA de las tres y ejecuten

# --- Opción 1 · EXCLUIR las filas sin cliente -------------------------------------------------
# limpio = limpio[~sin_cliente]

# --- Opción 2 · MARCAR la ausencia con una columna indicadora ---------------------------------
# limpio['sin_cliente'] = sin_cliente

# --- Opción 3 · «sin cliente» como CATEGORÍA PROPIA, en columna aparte -------------------------
# id_cliente queda intacto: la categoría no se escribe dentro de la columna decimal
# limpio['segmento_cliente'] = sin_cliente.map({True: 'sin cliente', False: 'identificado'})

print('Filas en el archivo de trabajo:', len(limpio))
print('Columnas:', list(limpio.columns))
"""),
    code("""
# SU LÍNEA DE BITÁCORA, en su propia celda como las cinco anteriores
# en `justificacion` descarten por escrito las dos opciones que NO tomaron, con su razón
#
# anotar(
#     variable='id_cliente',
#     proceso_generador='...',
#     decision='...',
#     justificacion='... Descarto excluir porque ... y la categoría propia porque ...',
#     efecto_esperado='...',
# )
"""),
    md("""
## Decisión 7 · Libre — elijan una de las tres

Las tres están sin resolver en el archivo. Cada una viene con **la evidencia mínima para decidir** y
con **la pregunta de negocio** que hay que responder. Elijan **una**, decidan y anótenla.

La evidencia no trae la respuesta: trae con qué sostenerla.

**Un aviso si en la decisión anterior eligieron excluir.** Las tres alternativas viven casi por
completo en los registros sin cliente, así que la evidencia de abajo se va a calcular sobre lo que quedó
y va a salir mucho más chica, o vacía. No es un error: es la decisión anterior moviendo la base. Si
les pasa, **eso mismo va en la bitácora** — es el efecto de su decisión, y observarlo vale tanto como
la decisión.
"""),
    md("""
### A · Cantidad negativa sin prefijo de cancelación

Las canceladas se marcan con `C` y todas tienen cantidad negativa. Pero hay registros negativos **sin**
esa marca.

**La pregunta de negocio: ¿son ventas?** Si lo son, restan del total vendido. Si no lo son, sumarlas
como ventas negativas deforma cualquier cifra de ventas.

Variables que hay que mirar: `cantidad`, `n_factura`, `id_cliente`, `precio_unitario` y lo que digan
las descripciones.
"""),
    code("""
neg_sin_c = (~limpio['n_factura'].str.startswith('C')) & (limpio['cantidad'] < 0)

print('Registros:', neg_sin_c.sum())
print()
"""),
    code("""
print(pd.DataFrame({
    'registros': [
        (neg_sin_c & limpio['id_cliente'].isna()).sum(),
        (neg_sin_c & (limpio['precio_unitario'] == 0)).sum(),
        (neg_sin_c & limpio['descripcion_producto'].notna()).sum(),
    ],
}, index=['sin cliente', 'con precio cero', 'con alguna descripción']).to_string())

"""),
    code("""
print('Qué dicen las que traen descripción:')
print(limpio.loc[neg_sin_c, 'descripcion_producto'].value_counts().head(6).to_string())
"""),
    md("""
### B · Líneas con precio cero

Un precio cero puede ser un dato que falta, un regalo o una muestra, o un ajuste que no es una venta.
Los tres se ven igual en la columna.

**La pregunta de negocio: ¿un precio cero significa «gratis» o significa «no se sabe»?** La respuesta
cambia por completo qué hacer con el registro.

Variables que hay que mirar: `precio_unitario`, `id_cliente` y `cantidad` —el signo separa dos
situaciones distintas—.
"""),
    code("""
precio_cero = limpio['precio_unitario'] == 0

print('Registros:', precio_cero.sum())
print()

"""),
    code("""
pd.DataFrame({
    'con cliente': [
        (precio_cero & limpio['id_cliente'].notna() & (limpio['cantidad'] > 0)).sum(),
        (precio_cero & limpio['id_cliente'].notna() & (limpio['cantidad'] < 0)).sum(),
    ],
    'sin cliente': [
        (precio_cero & limpio['id_cliente'].isna() & (limpio['cantidad'] > 0)).sum(),
        (precio_cero & limpio['id_cliente'].isna() & (limpio['cantidad'] < 0)).sum(),
    ],
}, index=['cantidad positiva', 'cantidad negativa'])
"""),
    md("""
### C · `pais`, donde `Unspecified` es un vacío disfrazado

`pais` no tiene ninguna celda vacía. Pero tiene categorías que **no nombran un país**.

**La pregunta de negocio: ¿es un país desconocido o es un país que no se registró?** Y una segunda,
que decide si hay algo que recuperar: **¿se puede deducir el país desde el cliente?**

Variables que hay que mirar: `pais` e `id_cliente`. La celda usa **`dropna()`** para quedarse solo
con los clientes que sí están identificados antes de buscarlos en el resto del archivo.
"""),
    code("""
sospechosos = ['Unspecified', 'European Community']
marca = limpio['pais'].isin(sospechosos)

print(limpio.loc[marca, 'pais'].value_counts().to_string())
print()
print('Clientes distintos en esos registros:', limpio.loc[marca, 'id_cliente'].nunique())

"""),
    code("""
# ¿Se puede recuperar el país? Solo si esos mismos clientes aparecen con un país concreto
clientes = limpio.loc[marca, 'id_cliente'].dropna().unique()
otras_filas = limpio[limpio['id_cliente'].isin(clientes) & ~marca]

print('Registros de esos mismos clientes con un país concreto:', len(otras_filas))
"""),
    md("""
### Cómo se aplica una decisión

Ya vieron cinco formas de corregir, y las cinco sirven acá. Elegir cuál corresponde **es** la
decisión; escribirla no debería costarles tiempo:

| Verbo | Se vio en | Se escribe así |
|:--|:--|:--|
| completar por regla | Decisión 1 | `limpio['col'] = limpio['col'].fillna(...)` |
| separar en dos columnas | Decisión 2 | `limpio['nueva'] = ...` |
| eliminar filas | Decisión 3 | `limpio = limpio[~mascara]` |
| normalizar en columna nueva | Decisión 4 | `limpio['col_norm'] = ...` |
| marcar con un indicador | Decisión 5 | `limpio['marca'] = mascara` |

Y la máscara de su alternativa **ya está calculada** por la celda de evidencia que acaban de
ejecutar: se llama `neg_sin_c` en la A, `precio_cero` en la B y `marca` en la C.
"""),
    code("""
# SU SEGUNDA DECISIÓN
# la máscara ya existe: neg_sin_c (A), precio_cero (B) o marca (C)
# apliquen su decisión sobre `limpio` con uno de los cinco verbos de la tabla

"""),
    code("""
# SU SEGUNDA LÍNEA DE BITÁCORA
"""),
    md("""
---

# Lo que quedó hecho

Las dos mitades, juntas.
"""),
    code("""
bitacora = pd.DataFrame(BITACORA)

print(f'Decisiones registradas: {len(BITACORA)} de {DECISIONES_ESPERADAS}')
print('Registros del archivo de trabajo:', len(limpio))

bitacora
"""),
    md("""
## El archivo corregido

La exportación **revisa primero que estén las siete decisiones**. Con menos, no escribe nada.

No es un capricho del cuaderno: un archivo corregido a medias parece terminado y no lo está, y quien
lo reciba no tiene cómo darse cuenta. La bitácora es lo que vuelve auditable al archivo, así que
salen juntos o no sale ninguno.

El dataset se escribe con **`to_parquet()`** y la bitácora con **`to_csv()`**: el primero conserva
los tipos para quien siga trabajando con el archivo, el segundo se abre en cualquier planilla.
"""),
    code("""
faltan = DECISIONES_ESPERADAS - len(BITACORA)

if faltan > 0:
    print(f'NO se exportó nada: faltan {faltan} decisiones de las {DECISIONES_ESPERADAS}.')
    print()
    print('Para completar la Parte 3:')
    print('  1. Decisión 6 · elijan una opción para id_cliente, aplíquenla y anótenla.')
    print('  2. Decisión 7 · elijan A, B o C, apliquen su decisión y anótenla.')
    print()
    print('Después vuelvan a ejecutar esta celda.')
else:
    # la bitácora se arma ACÁ y no antes: si se reusara la de la celda de vista previa, una
    # decisión agregada después quedaría fuera del archivo y nadie lo notaría
    bitacora = pd.DataFrame(BITACORA)

    limpio.to_parquet('ventas_online_limpio.parquet', index=False)
    bitacora.to_csv('bitacora_limpieza.csv', index=False)
    print(f'Escritos, con las {len(BITACORA)} decisiones registradas:')
    print('  ventas_online_limpio.parquet · bitacora_limpieza.csv')
"""),
    md("""
## Lo que queda para después

Tres cosas que **no** se hicieron hoy, a propósito:

- **Escalar, codificar y crear variables**: eso es la clase 5. Hoy se corrigió lo que estaba mal, no
  se transformó lo que ya estaba bien.
- **Separar entrenamiento y prueba**: es la clase 6.
- **Medir un modelo**: también la clase 6. Hoy no se midió nada.

Una regla que empieza a regir **desde ahora**, aunque su porqué llegue después: todo lo que
**aprende un parámetro de los datos** —el valor con que se imputa, el umbral con que se recorta— se
calcula **solo con los datos de entrenamiento**. Corregir un tipo, separar una columna o eliminar
una repetición identificada no aprende nada, así que puede hacerse sobre el archivo completo. Imputar
y recortar, no.
"""),
    md("""
## Atribución de datos

- **Creador:** Chen, D. (2012)
- **Fuente:** [UCI Machine Learning Repository — *Online Retail II*, dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Modificación:** las ocho columnas se renombraron al español en `snake_case`. Los valores no se
  tocaron: llegan en inglés, tal como están en el original.
"""),
]


def main() -> None:
    nb = {
        "cells": con_ids(CELDAS),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    SALIDA.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md_n = sum(1 for c in CELDAS if c["cell_type"] == "markdown")
    print(f"escrito {SALIDA.relative_to(REPO_ROOT)}")
    print(f"  {len(CELDAS)} celdas · {md_n} markdown · {len(CELDAS) - md_n} código")


if __name__ == "__main__":
    main()
