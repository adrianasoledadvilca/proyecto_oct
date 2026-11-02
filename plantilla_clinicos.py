import pandas as pd

ids = pd.read_csv("ids_autorizacion.csv")[["id_paciente"]]
ids["bloque_diagnostico"] = ""   # ej. H35 (primeras 3 posiciones del CIE-10)
ids["edad"] = ""                 # años

ids.to_csv("datos_clinicos.csv", index=False)
print(f"Plantilla creada con {len(ids)} filas: completala en Excel y guardala como CSV.")