@echo off
cd /d "%~dp0"

set PORT=7734
set URL=http://localhost:%PORT%

echo ==============================
echo   Starting policy server...
echo ==============================

:: Start server in background
start /b python server.py

echo Waiting for server to be ready...

:WAIT_LOOP
timeout /t 1 /nobreak > nul
curl -s %URL% > nul 2>&1
if errorlevel 1 goto WAIT_LOOP

echo Server is up! Opening browser...
start %URL%

echo.
echo Server running at %URL%
echo Close this window to stop the server.

:: Keep window open
pause
