@echo off
chcp 65001 >nul
cd /d "%~dp0"
call D:\miniconda3\Scripts\activate.bat D:\conda_envs\python312
python main.py ui
pause
