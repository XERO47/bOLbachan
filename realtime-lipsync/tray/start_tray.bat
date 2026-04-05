@echo off
cd /d "%~dp0"

REM Server URL and camera are set in config.json
REM Run install.bat first if you haven't already.
REM Add --start to auto-start streaming on launch.

python tray.py --start %*
