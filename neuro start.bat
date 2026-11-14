@echo off
cd /d D:\ml\NEUROSYNTARA-main
call venv\Scripts\activate.bat
cd NEUROSYNTARA-main
python -m uvicorn main:app --reload
pause
