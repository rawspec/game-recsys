@echo off
cd /d %~dp0
.venv\Scripts\python -m app.server --open --port 8000
pause
