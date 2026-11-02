from pathlib import Path

CARPETAS = [
    "expedientes/AUTORIZADO",
    "expedientes/NO_AUTORIZADO",
    "paginas_todas",
    "paginas_etiquetadas/topografia_corneal",
    "paginas_etiquetadas/modulo_81",
    "paginas_etiquetadas/paquimetria",
    "paginas_etiquetadas/campimetria",
    "paginas_etiquetadas/ultima_oct",
    "paginas_etiquetadas/recuento_celulas",
    "paginas_etiquetadas/pedido",
    "paginas_etiquetadas/otro",
]

for carpeta in CARPETAS:
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    print("Lista:", carpeta)