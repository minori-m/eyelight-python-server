@echo off
REM ========================
REM 設定
REM ========================
set PROJECT_DIR=C:\Users\pupil2\Documents\pupil
set VENV_DIR=%PROJECT_DIR%\venv
set PY_SCRIPT=%PROJECT_DIR%\unite_full.py

set PUPIL_EXE="C:\Program Files (x86)\Pupil-Labs\Pupil v3.5.1\Pupil Capture v3.5.1\pupil_capture.exe"

REM ========================
REM 作業ディレクトリ
REM ========================
cd /d %PROJECT_DIR%

REM ========================
REM venv 有効化
REM ========================
call %VENV_DIR%\Scripts\activate.bat

REM ========================
REM Pupil Capture 起動
REM ========================
start "" %PUPIL_EXE%

REM ========================
REM 起動待ち（秒）
REM ========================
timeout /t 8 /nobreak

REM ========================
REM Python 実行
REM ========================
python %PY_SCRIPT%

pause
