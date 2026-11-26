@echo off
setlocal
cd /d "%~dp0"
echo Starting the Routewise Python server...
echo When ready, open http://127.0.0.1:8000 in your browser.
echo Press Ctrl+C here to stop the server.
python app.py
if errorlevel 1 (
    echo.
    echo The server could not start. Check that Python is installed and port 8000 is free.
    pause
)
