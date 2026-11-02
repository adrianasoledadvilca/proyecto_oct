import pandas as pd

clinicos = pd.read_csv("datos_clinicos.csv")
ids = pd.read_csv("ids_autorizacion.csv")

print("Filas en datos_clinicos.csv:", len(clinicos))
print("Filas en ids_autorizacion.csv:", len(ids))
print("Celdas vacías por columna:")
print(clinicos.isna().sum())
print("IDs distintos entre ambos archivos:", set(clinicos["id_paciente"]) ^ set(ids["id_paciente"]))
print("Bloques de diagnóstico cargados:")
print(clinicos["bloque_diagnostico"].value_counts())