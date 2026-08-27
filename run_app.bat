@echo off
echo ========================================================
echo   Plaka Tanima Sistemi (UI) Baslatiliyor...
echo ========================================================
echo.

cd /d "%~dp0PlakaOkuma-NumberPlateRecognition"

if exist "..\venv\Scripts\python.exe" (
    echo [OK] venv ortaminda baslatiliyor...
    ..\venv\Scripts\python.exe ui.py
) else (
    echo [UYARI] Sanal ortam bulunamadi, varsayilan python ile deneniyor...
    python ui.py
)

pause
