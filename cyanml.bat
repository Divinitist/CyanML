@echo off
setlocal
set "CYANML_ROOT=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CYANML_ROOT%scripts\cyanml.ps1" %*
exit /b %ERRORLEVEL%
