@echo off
cd /d "%~dp0"

REM Edit SERVER to point to your GPU instance
set SERVER=ws://205.147.102.96:8000/ws/deepfake

REM Camera index (0 = default webcam, 1 = second cam, etc.)
set CAM=0

REM Modes (uncomment one):
REM   Normal window (OBS Window Capture):
python viewer.py --server %SERVER% --cam %CAM% %*
REM   Headless — OBS uses Media Source URL instead:
REM python viewer.py --server %SERVER% --cam %CAM% --no-window %*

pause
