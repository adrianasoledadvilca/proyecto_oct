from pathlib import Path
import joblib
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. RECONOCIMIENTO DE IMÁGENES: qué estudios tiene cada expediente ---
guardado = torch.load("resnet18_estudios.pth")
clases = guardado["clases"]
resnet = models.resnet18()
resnet.fc = nn.Linear(resnet.fc.in_features, len(clases))
resnet.load_state_dict(guardado["pesos"])
resnet.eval()

transformar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
ESTUDIOS = [c for c in clases if c not in ("otro", "pedido")]

filas = {}
for ruta in sorted(Path("paginas_todas").glob("*.png")):
    id_paciente = ruta.stem.rsplit("_p", 1)[0]
    imagen = transformar(Image.open(ruta).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        tipo = clases[resnet(imagen).argmax(dim=1).item()]
    fila = filas.setdefault(id_paciente, {"id_paciente": id_paciente, **{e: 0 for e in ESTUDIOS}})
    if tipo in ESTUDIOS:
        fila[tipo] = 1

estudios = pd.DataFrame(filas.values())
estudios["cantidad_estudios"] = estudios[ESTUDIOS].sum(axis=1)

# --- 2. UNIR CON DATOS CLÍNICOS Y OBJETIVO ---
clinicos = pd.read_csv("datos_clinicos.csv")
objetivo = pd.read_csv("ids_autorizacion.csv")
base = estudios.merge(clinicos, on="id_paciente").merge(objetivo, on="id_paciente")
base.to_csv("base_estructurada.csv", index=False)
print("Expedientes en la base:", len(base))
print(base["autorizacion"].value_counts())

# --- 3. CLASIFICADOR: RANDOM FOREST ---
numericas = ESTUDIOS + ["cantidad_estudios", "edad"]
categoricas = ["bloque_diagnostico"]
X = base[numericas + categoricas]
y = base["autorizacion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = Pipeline([
    ("preparacion", ColumnTransformer(
        [("categorias", OneHotEncoder(handle_unknown="ignore"), categoricas)],
        remainder="passthrough")),
    ("bosque", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
])
modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
print("Exactitud entrenamiento:", round(modelo.score(X_train, y_train), 3))
print("Exactitud prueba:", round(modelo.score(X_test, y_test), 3))
print(confusion_matrix(y_test, predicciones, labels=modelo.classes_))
print(classification_report(y_test, predicciones, zero_division=0))

nombres = modelo.named_steps["preparacion"].get_feature_names_out()
importancias = pd.Series(modelo.named_steps["bosque"].feature_importances_, index=nombres)
print(importancias.sort_values(ascending=False).head(10).round(3))

# --- 4. GUARDAR VERSIÓN DEL MODELO ---
import json
import shutil
from datetime import datetime

version = datetime.now().strftime("v%Y%m%d_%H%M%S")
carpeta = Path("modelos") / version
carpeta.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "version": version,
    "modelo": modelo,
    "columnas": numericas + categoricas,
    "estudios": ESTUDIOS,
    "n_entrenamiento": len(base),
}, carpeta / "modelo_autorizacion.joblib")
shutil.copy(carpeta / "modelo_autorizacion.joblib", "modelo_autorizacion.joblib")

metricas = {
    "version": version,
    "fecha": datetime.now().isoformat(timespec="seconds"),
    "expedientes_total": len(base),
    "expedientes_entrenamiento": len(X_train),
    "expedientes_prueba": len(X_test),
    "exactitud_entrenamiento": round(modelo.score(X_train, y_train), 3),
    "exactitud_prueba": round(modelo.score(X_test, y_test), 3),
}
(carpeta / "metricas.json").write_text(
    json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Modelo guardado como {version} en la carpeta modelos/")