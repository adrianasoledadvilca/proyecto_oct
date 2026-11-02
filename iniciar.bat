@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
echo Abrí http://localhost:8000 en el navegador (Ctrl+C para detener)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause