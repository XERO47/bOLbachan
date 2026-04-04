@echo off
cd /d "%~dp0"

REM Edit SERVER to point to your GPU instance
set SERVER=ws://205.147.101.238:8000/ws/deepfake

REM Camera index (0 = default webcam, 1 = second cam, etc.)
set CAM=0

REM Add --vcam to also push to OBS Virtual Camera device
python viewer.py --server %SERVER% --cam %CAM% %*

pause
