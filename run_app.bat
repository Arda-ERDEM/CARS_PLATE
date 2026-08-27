@echo off
title Plaka Tanima Sistemi
echo ========================================================
echo   Plaka Tanima Sistemi (UI) Baslatiliyor...
echo ========================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [BILGI] Ilk calistirma algilandi. Gerekli kurulumlar yapiliyor...
    echo Lutfen bekleyin, kutuphanelerin inmesi birkac dakika surebilir.
    echo.
    echo [1/3] Sanal ortam (venv) olusturuluyor...
    python -m venv venv
    
    echo [2/3] Pip guncelleniyor...
    venv\Scripts\python.exe -m pip install --upgrade pip
    
    echo [3/3] Kutuphaneler (requirements.txt) yukleniyor...
    venv\Scripts\pip.exe install -r requirements.txt
    
    echo.
    echo [OK] Tum kurulumlar basariyla tamamlandi!
    echo.
)

cd PlakaOkuma-NumberPlateRecognition

echo [OK] Uygulama baslatiliyor...
..\venv\Scripts\python.exe ui.py

pause
