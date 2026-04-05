@echo off
echo =============================================
echo  DeepFace Tray - Windows Setup
echo =============================================
echo.

echo [1/3] Installing Python packages...
pip install pystray pillow opencv-python websockets numpy pyvirtualcam
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking OBS Virtual Camera driver...
reg query "HKLM\SOFTWARE\obs-studio" >nul 2>&1
if errorlevel 1 (
    echo WARNING: OBS Studio not detected.
    echo.
    echo You need the OBS Virtual Camera driver for Meet/Zoom/Teams.
    echo Install OBS from https://obsproject.com, open it once,
    echo click 'Start Virtual Camera', then close it.
    echo The driver stays installed permanently.
    echo.
) else (
    echo OBS detected - virtual camera driver should be available.
)

echo [3/3] Creating config if not exists...
if not exist "%~dp0config.json" (
    echo {"server": "ws://205.147.102.96:8000/ws/deepfake", "cam": 0, "fps": 25, "quality": 85} > "%~dp0config.json"
    echo Config created: config.json
) else (
    echo Config already exists: config.json
)

echo.
echo =============================================
echo  Done! Edit config.json to set your server.
echo  Then run: start_tray.bat
echo =============================================
pause
