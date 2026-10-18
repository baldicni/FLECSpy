
@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   Setup ambiente Python
echo ========================================

where py >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python non trovato.
    echo Installa Python da https://www.python.org/downloads/windows/
    echo e seleziona Add Python to PATH.
    pause
    exit /b 1
)

echo.
echo [1/3] Creazione ambiente virtuale...

if not exist ".venv\Scripts\python.exe" (
    py -m venv .venv
    if errorlevel 1 goto error
) else (
    echo Ambiente .venv gia' presente.
)

echo.
echo [2/3] Aggiornamento pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error

echo.
echo [3/3] Installazione librerie...
".venv\Scripts\python.exe" -m pip install numpy sympy scipy matplotlib
if errorlevel 1 goto error

echo.
echo Installazione completata!
echo Per attivare l'ambiente esegui:
echo .venv\Scripts\activate.bat
echo.
pause
exit /b 0

:error
echo.
echo ERRORE durante la configurazione.
echo Controlla la connessione internet e riprova.
pause
exit /b 1
