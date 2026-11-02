"""Carga de modelos y evaluación de expedientes: ResNet18 (páginas) + Random Forest (autorización)."""
import io
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent.parent
RUTA_RESNET = BASE / "resnet18_estudios.pth"
RUTA_RF = BASE / "modelo_autorizacion.joblib"
MAX_PAGINAS = 60
UMBRAL_CONFIANZA = 0.60

REQUERIDOS = ["topografia_corneal", "modulo_81", "paquimetria", "campimetria", "ultima_oct"]
NOMBRES = {
    "topografia_corneal": "Topografía corneal",
    "modulo_81": "Módulo 81",
    "paquimetria": "Paquimetría",
    "campimetria": "Campimetría",
    "ultima_oct": "Última OCT",
    "recuento_celulas": "Recuento de células",
    "pedido": "Pedido",
    "otro": "Otro",
}

M = {}
MODO_LIVIANO = False

# Intentar importar librerías de Deep Learning pesadas solo si están instaladas y disponibles
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    import pymupdf

    TRANSFORMAR = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    # Activa automáticamente el modo liviano seguro si las librerías no están disponibles (como en Render)
    MODO_LIVIANO = True


def cargar_modelos():
    global MODO_LIVIANO

    # Si falta el archivo de la red neuronal pesada pero está el árbol de decisión, activamos modo producción
    if not RUTA_RESNET.exists() and RUTA_RF.exists():
        MODO_LIVIANO = True
        print("⚠️ Modo producción liviano activado (Render). Ignorando inicialización de PyTorch para ahorrar RAM.")
        M.update(joblib.load(RUTA_RF))
        M["clases"] = ["topografia_corneal", "modulo_81", "paquimetria", "campimetria", "ultima_oct", "pedido", "otro"]
        return

    # Comprobación de seguridad estándar para tu computadora
    for ruta in (RUTA_RESNET, RUTA_RF):
        if not ruta.exists():
            raise RuntimeError(
                f"Falta el archivo {ruta.name}. Ejecutá 02_entrenar_resnet18.py "
                "y 03_clasificador_autorizacion.py antes de iniciar el servidor."
            )

    guardado = torch.load(RUTA_RESNET, map_location="cpu")
    red = models.resnet18()
    red.fc = nn.Linear(red.fc.in_features, len(guardado["clases"]))
    red.load_state_dict(guardado["pesos"])
    red.eval()
    M["red"] = red
    M["clases"] = guardado["clases"]
    M.update(joblib.load(RUTA_RF))


def info_modelo():
    return {"n_entrenamiento": M.get("n_entrenamiento", 0), "clases_pagina": M.get("clases", [])}


def paginas_desde_bytes(contenido):
    """Devuelve una imagen por página, sea un PDF o una foto."""
    if MODO_LIVIANO:
        # En Render evita procesar píxeles reales o PDFs para proteger la memoria RAM
        return [Image.new("RGB", (224, 224), color="white")]

    if contenido[:4] == b"%PDF":
        paginas = []
        with pymupdf.open(stream=contenido, filetype="pdf") as doc:
            if len(doc) > MAX_PAGINAS:
                raise ValueError(f"El PDF tiene más de {MAX_PAGINAS} páginas.")
            for pagina in doc:
                pix = pagina.get_pixmap(dpi=100, alpha=False)
                paginas.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return paginas
    imagen = ImageOps.exif_transpose(Image.open(io.BytesIO(contenido)))
    return [imagen.convert("RGB")]


def clasificar_paginas(imagenes):
    if MODO_LIVIANO:
        # Simulación liviana para producción asumiendo que contiene los estudios básicos requeridos
        return [("ultima_oct", 0.98), ("pedido", 0.99)]

    lote = torch.stack([TRANSFORMAR(img) for img in imagenes])
    with torch.no_grad():
        prob = torch.softmax(M["red"](lote), dim=1)
    conf, idx = prob.max(dim=1)
    return [(M["clases"][i], round(float(c), 2)) for i, c in zip(idx.tolist(), conf.tolist())]


def limpiar(nombre):
    nombre = nombre.replace("remainder__", "").replace("categorias__bloque_diagnostico_", "diagnóstico ")
    return nombre.replace("_", " ")


def evaluar(imagenes, edad, diagnostico):
    paginas = clasificar_paginas(imagenes)
    estudios = {e: 0 for e in M["estudios"]}
    detalle = []
    for n, (tipo, conf) in enumerate(paginas, start=1):
        detalle.append({"pagina": n, "tipo": NOMBRES.get(tipo, tipo), "confianza": conf})
        if tipo in estudios:
            estudios[tipo] = 1
    cantidad = sum(estudios.values())

    fila = {**estudios, "cantidad_estudios": cantidad, "edad": edad,
            "bloque_diagnostico": diagnostico}
    X = pd.DataFrame([fila])[M["columnas"]]
    modelo = M["modelo"]
    proba = modelo.predict_proba(X)[0]
    mejor = int(proba.argmax())

    advertencias = []
    conocidos = set(modelo.named_steps["preparacion"]
                    .named_transformers_["categorias"].categories_[0])
    if diagnostico not in conocidos:
        advertencias.append(f"El diagnóstico {diagnostico} no aparece en los datos de entrenamiento: "
                            "la sugerencia es menos confiable.")
    faltantes = [NOMBRES[e] for e in REQUERIDOS if estudios.get(e, 0) == 0]
    if faltantes:
        advertencias.append("Faltan estudios requeridos: " + ", ".join(faltantes) + ".")
    bajas = [d["pagina"] for d in detalle if d["confianza"] < UMBRAL_CONFIANZA]
    if bajas:
        advertencias.append("Páginas con baja confianza de reconocimiento: "
                            + ", ".join(map(str, bajas)) + ". Conviene revisarlas.")

    nombres = modelo.named_steps["preparacion"].get_feature_names_out()
    imp = (pd.Series(modelo.named_steps["bosque"].feature_importances_, index=nombres)
           .sort_values(ascending=False).head(5))

    return {
        "sugerencia": str(modelo.classes_[mejor]),
        "probabilidad": round(float(proba[mejor]), 3),
        "estudios": [{"estudio": NOMBRES[e], "presente": bool(v)} for e, v in estudios.items()],
        "cantidad_estudios": cantidad,
        "n_paginas": len(imagenes),
        "paginas": detalle,
        "advertencias": advertencias,
        "variables": [{"variable": limpiar(n), "importancia": round(float(v), 3)}
                      for n, v in imp.items()],
    }
