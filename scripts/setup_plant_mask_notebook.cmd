@echo off
REM Wrapper for setup_plant_mask_notebook.ps1 (avoids PowerShell execution-policy errors)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_plant_mask_notebook.ps1" %*
if errorlevel 1 exit /b %errorlevel%
