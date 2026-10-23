@echo off
rem Double-click to read your books in the browser (same as `tbook open`).
rem The minimized "AutoBook Reader" window is the reader's local server: close it to stop.
if not "%~1"=="--minimized" (
  start "AutoBook Reader" /min "%~f0" --minimized
  exit /b
)
cd /d "%~dp0"
".venv\Scripts\tbook.exe" open
if errorlevel 1 pause
