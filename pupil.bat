@echo off
cd /d "%~dp0"

rem ① Pupil Capture 起動
start "" "C:\Program Files (x86)\Pupil-Labs\Pupil v3.5.1\Pupil Capture v3.5.1\pupil_capture.exe"

rem ② 少し待つ（起動待ち）
timeout /t 5

rem ③ venv Python で unite 起動
venv\Scripts\python.exe unite_full.py

cmd /k
