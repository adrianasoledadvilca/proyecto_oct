import shutil
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

ORIGEN = Path("paginas_todas")
DESTINO = Path("paginas_etiquetadas")
CLASES = {
    "1": "topografia_corneal",
    "2": "modulo_81",
    "3": "paquimetria",
    "4": "campimetria",
    "5": "ultima_oct",
    "6": "recuento_celulas",
    "7": "pedido",
    "8": "otro",
}

for carpeta in CLASES.values():
    (DESTINO / carpeta).mkdir(parents=True, exist_ok=True)

ya_etiquetadas = {p.name for c in CLASES.values() for p in (DESTINO / c).glob("*.png")}
pendientes = [p for p in sorted(ORIGEN.glob("*.png")) if p.name not in ya_etiquetadas]
print(f"Pendientes: {len(pendientes)} | Ya etiquetadas: {len(ya_etiquetadas)}")

ventana = tk.Tk()
ventana.title("Etiquetar páginas")
ventana.geometry("900x1000")

texto = tk.Label(ventana, font=("Arial", 12), justify="left")
texto.pack(pady=5)
lienzo = tk.Label(ventana)
lienzo.pack(expand=True)

indice = 0
foto = None


def mostrar():
    global foto
    if indice >= len(pendientes):
        texto.config(text="¡Listo! No quedan páginas pendientes.")
        lienzo.config(image="")
        return
    ruta = pendientes[indice]
    imagen = Image.open(ruta)
    imagen.thumbnail((850, 900))
    foto = ImageTk.PhotoImage(imagen)
    lienzo.config(image=foto)
    leyenda = "   ".join(f"{k}={v}" for k, v in CLASES.items())
    texto.config(text=f"[{indice + 1}/{len(pendientes)}] {ruta.name}\n{leyenda}   S=saltear   Q=salir")


def tecla(evento):
    global indice
    t = evento.char.lower()
    if t == "q":
        ventana.destroy()
        return
    if indice >= len(pendientes):
        return
    if t in CLASES:
        origen = pendientes[indice]
        shutil.copy(origen, DESTINO / CLASES[t] / origen.name)
        indice += 1
        mostrar()
    elif t == "s":
        indice += 1
        mostrar()


ventana.bind("<Key>", tecla)
mostrar()
ventana.mainloop()