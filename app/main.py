"""Servidor FastAPI: página, análisis, decisiones, historial, alertas, CIE-10 y métricas."""
import json
import re
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import db, pipeline

BASE = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 25 * 1024 * 1024
PRACTICAS = {"OCT", "Inyección de Vabysmo", "Cirugía de cataratas", "Sin clasificar"}
DECISIONES = {"AUTORIZADO", "NO_AUTORIZADO", "PEDIR_ESTUDIOS"}
CIE10 = json.loads((Path(__file__).resolve().parent / "cie10.json").read_text(encoding="utf-8"))


def _sin_acentos(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


@asynccontextmanager
async def lifespan(app):
    db.iniciar()
    pipeline.cargar_modelos()
    yield


app = FastAPI(title="Evaluación de estudios OCT con machine learning", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def inicio():
    return FileResponse(STATIC / "index.html")


@app.get("/api/salud")
def salud():
    return {"estado": "ok", "version_modelo": pipeline.M.get("version", "sin versión"),
            **pipeline.info_modelo()}


@app.post("/api/evaluar")
async def evaluar(
    archivos: List[UploadFile] = File(...),
    edad: int = Form(...),
    diagnostico: str = Form(...),
    practica: str = Form(...),
):
    dx = diagnostico.strip().upper()
    if not re.fullmatch(r"[A-Z][0-9]{2,3}", dx):
        raise HTTPException(422, "El diagnóstico debe tener el formato H353.")
    if not 0 <= edad <= 120:
        raise HTTPException(422, "La edad debe estar entre 0 y 120.")
    if practica not in PRACTICAS:
        raise HTTPException(422, "Práctica no válida.")

    imagenes = []
    for archivo in archivos:
        contenido = await archivo.read()
        if len(contenido) > MAX_BYTES:
            raise HTTPException(413, f"«{archivo.filename}» supera los 25 MB.")
        try:
            imagenes += await run_in_threadpool(pipeline.paginas_desde_bytes, contenido)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            raise HTTPException(400, f"No se pudo leer «{archivo.filename}». ¿Es PDF o imagen?")

    if not imagenes:
        raise HTTPException(400, "No se recibieron páginas.")
    if len(imagenes) > pipeline.MAX_PAGINAS:
        raise HTTPException(400, f"Máximo {pipeline.MAX_PAGINAS} páginas por expediente.")

    resultado = await run_in_threadpool(pipeline.evaluar, imagenes, edad, dx)
    version = pipeline.M.get("version", "sin versión")
    nuevo_id = db.guardar(edad, dx, practica, resultado, version)
    resultado.update(id_expediente=nuevo_id, diagnostico=dx, practica=practica,
                     version_modelo=version)
    return resultado


@app.post("/api/decision")
def decision(id_expediente: int = Form(...), decision: str = Form(...), motivo: str = Form("")):
    if decision not in DECISIONES:
        raise HTTPException(422, "Decisión no válida.")
    if not db.registrar_decision(id_expediente, decision, motivo.strip()[:300]):
        raise HTTPException(404, "No existe ese expediente.")
    return {"ok": True}


@app.get("/api/historial")
def historial():
    return db.historial()


@app.get("/api/similares/{id_expediente}")
def similares(id_expediente: int):
    exp = db.obtener(id_expediente)
    if not exp:
        raise HTTPException(404, "No existe ese expediente.")
    return db.similares(id_expediente, exp["diagnostico"], exp["practica"])


@app.get("/api/alertas")
def alertas(dias: int = 3):
    return db.sin_respuesta(dias)


@app.get("/api/cie10")
def buscar_cie10(q: str = ""):
    texto = _sin_acentos(q.strip())
    if not texto:
        return CIE10[:20]
    return [c for c in CIE10
            if texto in _sin_acentos(c["codigo"]) or texto in _sin_acentos(c["descripcion"])][:20]


@app.get("/api/metricas")
def metricas():
    versiones = []
    for m in sorted((BASE / "modelos").glob("*/metricas.json"), reverse=True):
        versiones.append(json.loads(m.read_text(encoding="utf-8")))
    return {"versiones": versiones, "decisiones": db.concordancia()}