"""Base SQLite: evaluaciones, decisiones de la médica y métricas (solo datos pseudonimizados)."""
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

RUTA_PREDETERMINADA = Path(__file__).resolve().parent.parent / "datos" / "expedientes.db"
RUTA = Path(os.environ.get("DB_PATH", str(RUTA_PREDETERMINADA)))


def _con():
    RUTA.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(RUTA)
    con.row_factory = sqlite3.Row
    return con


def iniciar():
    con = _con()
    with con:
        con.execute("""CREATE TABLE IF NOT EXISTS evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            edad INTEGER NOT NULL,
            diagnostico TEXT NOT NULL,
            practica TEXT NOT NULL,
            estudios TEXT NOT NULL,
            sugerencia TEXT NOT NULL,
            probabilidad REAL NOT NULL,
            version_modelo TEXT NOT NULL,
            decision TEXT,
            fecha_decision TEXT,
            motivo TEXT)""")
    con.close()


def guardar(edad, diagnostico, practica, resultado, version):
    con = _con()
    with con:
        cur = con.execute(
            """INSERT INTO evaluaciones (fecha, edad, diagnostico, practica, estudios,
               sugerencia, probabilidad, version_modelo)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec="seconds"), edad, diagnostico, practica,
             json.dumps(resultado["estudios"], ensure_ascii=False),
             resultado["sugerencia"], resultado["probabilidad"], version),
        )
    nuevo = cur.lastrowid
    con.close()
    return nuevo


def obtener(id_exp):
    con = _con()
    fila = con.execute("SELECT * FROM evaluaciones WHERE id = ?", (id_exp,)).fetchone()
    con.close()
    return dict(fila) if fila else None


def registrar_decision(id_exp, decision, motivo):
    con = _con()
    with con:
        cur = con.execute(
            "UPDATE evaluaciones SET decision = ?, fecha_decision = ?, motivo = ? WHERE id = ?",
            (decision, datetime.now().isoformat(timespec="seconds"), motivo, id_exp),
        )
    filas = cur.rowcount
    con.close()
    return filas > 0


def similares(id_exp, diagnostico, practica, limite=10):
    con = _con()
    filas = con.execute(
        """SELECT id, fecha, diagnostico, practica, sugerencia, decision
           FROM evaluaciones
           WHERE id != ? AND decision IS NOT NULL
             AND (diagnostico = ? OR practica = ?)
           ORDER BY id DESC LIMIT ?""",
        (id_exp, diagnostico, practica, limite),
    ).fetchall()
    con.close()
    return [dict(f) for f in filas]


def sin_respuesta(dias=3):
    corte = (datetime.now() - timedelta(days=dias)).isoformat(timespec="seconds")
    con = _con()
    filas = con.execute(
        """SELECT id, fecha, diagnostico, practica, sugerencia FROM evaluaciones
           WHERE decision IS NULL AND fecha < ? ORDER BY fecha""",
        (corte,),
    ).fetchall()
    con.close()
    return [dict(f) for f in filas]


def concordancia():
    con = _con()
    fila = con.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN sugerencia = decision THEN 1 ELSE 0 END) AS coinciden
           FROM evaluaciones WHERE decision IN ('AUTORIZADO', 'NO_AUTORIZADO')"""
    ).fetchone()
    con.close()
    total, coinciden = fila["total"] or 0, fila["coinciden"] or 0
    return {
        "decididos": total,
        "coinciden": coinciden,
        "concordancia": round(coinciden / total, 3) if total else None,
    }
def historial(limite=200):
    con = _con()
    filas = con.execute(
        """SELECT id, fecha, edad, diagnostico, practica, estudios, sugerencia,
                  probabilidad, version_modelo, decision, fecha_decision, motivo
           FROM evaluaciones ORDER BY id DESC LIMIT ?""",
        (limite,),
    ).fetchall()
    con.close()
    return [dict(f) for f in filas]