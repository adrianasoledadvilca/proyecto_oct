from pathlib import Path
import pymupdf as fitz  # PyMuPDF
import pandas as pd

CARPETA_PDFS = Path("expedientes")
SALIDA = Path("paginas_todas")
SALIDA.mkdir(exist_ok=True)

filas = []
contador = 0
for resultado in ["AUTORIZADO", "NO_AUTORIZADO"]:
    for pdf in sorted((CARPETA_PDFS / resultado).glob("*.pdf")):
        contador += 1
        id_paciente = f"ID_Paciente_{contador:03d}"  # sin nombres ni CUIL
        with fitz.open(pdf) as documento:
            for numero, pagina in enumerate(documento, start=1):
                pagina.get_pixmap(dpi=100).save(SALIDA / f"{id_paciente}_p{numero}.png")
        filas.append({"id_paciente": id_paciente, "autorizacion": resultado})

pd.DataFrame(filas).to_csv("ids_autorizacion.csv", index=False)
print(f"{contador} expedientes convertidos.")