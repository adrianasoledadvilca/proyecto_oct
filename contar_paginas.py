from pathlib import Path

for carpeta in sorted(Path("paginas_etiquetadas").iterdir()):
    if carpeta.is_dir():
        print(f"{carpeta.name}: {len(list(carpeta.glob('*.png')))} páginas")