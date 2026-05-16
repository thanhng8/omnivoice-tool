@echo off
setlocal enabledelayedexpansion
title OmniVoice TTS Server

REM ============================================================
REM  Offline mode for HuggingFace
REM  After first download, models are cached at
REM      %USERPROFILE%\.cache\huggingface\hub
REM  HF_HUB_OFFLINE=1 skips network checks for faster startup.
REM  Comment out the lines below or delete the cache folder if
REM  you want to re-download / update the model.
REM ============================================================
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

REM ============================================================
REM  Locate Python where omnivoice is installed
REM ============================================================
where /q omnivoice-infer
if not errorlevel 1 (
    for /f "delims=" %%i in ('python -c "import sys, omnivoice; print(sys.executable)"') do set PY=%%i
) else (
    set PY=python
)

REM ============================================================
REM  If args were passed directly, skip menu
REM ============================================================
if not "%~1"=="" (
    set "ARGS=%*"
    goto :run
)

:menu
cls
echo ============================================================
echo   OmniVoice TTS Server - Launcher
echo ============================================================
echo.
echo   Web UI:  http://127.0.0.1:8765/  ^(default port^)
echo   Python:  %PY%
echo.
echo   Choose a launch mode:
echo.
echo     [1] GPU + Whisper ASR  ^(default - uses ~4GB VRAM^)
echo         Allows uploading voice profiles without ref_text
echo         ^(Whisper auto-transcribes the reference audio^)
echo.
echo     [2] GPU only ^(no Whisper^) - saves 1.5GB VRAM
echo         ref_text is required when uploading voice profiles
echo.
echo     [3] CPU only ^(slow - for machines without a GPU^)
echo.
echo     [4] GPU + Whisper, custom port
echo.
echo     [5] Custom args ^(advanced^)
echo.
echo     [Q] Quit
echo.
set /p CHOICE=Your choice [1/2/3/4/5/Q]:

if /i "%CHOICE%"=="1" (
    set "ARGS="
    goto :run
)
if /i "%CHOICE%"=="2" (
    set "ARGS=--no-asr"
    goto :run
)
if /i "%CHOICE%"=="3" (
    set "ARGS=--cpu"
    goto :run
)
if /i "%CHOICE%"=="4" (
    set /p PORT=Enter port [e.g. 8888]:
    if "!PORT!"=="" set PORT=8765
    set "ARGS=--port !PORT!"
    goto :run
)
if /i "%CHOICE%"=="5" (
    echo.
    echo Common flags:
    echo   --port N        Change port ^(default 8765^)
    echo   --no-asr        Skip Whisper ASR
    echo   --cpu           Force CPU
    echo   --device cuda^|mps^|cpu   Pick a device
    echo   --dtype float16^|float32^|bfloat16
    echo.
    set /p CUSTOM=Enter args:
    set "ARGS=!CUSTOM!"
    goto :run
)
if /i "%CHOICE%"=="Q" (
    exit /b 0
)
echo.
echo Invalid choice. Try again...
timeout /t 2 >nul
goto :menu

:run
cls
echo ============================================================
echo   OmniVoice TTS Server (HTTP + WebSocket)
echo ============================================================
echo.
echo   Args:           %ARGS%
echo   Python:         %PY%
echo   HF_HUB_OFFLINE: %HF_HUB_OFFLINE%
echo.
echo   Web UI:         http://127.0.0.1:8765/
echo   WebSocket:      ws://127.0.0.1:8765/ws
echo   Voice profiles: %~dp0voice_prompts\
echo.
echo   Press Ctrl+C to stop the server. Closing this window also stops it.
echo.
echo ============================================================
echo.

"%PY%" "%~dp0ws_omnivoice_server.py" %ARGS%

echo.
echo Server stopped. Press any key to exit...
pause >nul
