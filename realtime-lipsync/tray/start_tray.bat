@echo off
cd /d "%~dp0"

REM Server WebSocket URL — update after running install_services.sh on the GPU server
set SERVER=ws://205.147.102.96:8000/ws/deepfake

REM Camera index (0 = default webcam)
set CAM=0

REM Remove --start to NOT auto-start streaming on launch
python tray.py --server %SERVER% --cam %CAM% --start %*
