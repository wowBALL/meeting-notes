@echo off
chcp 65001 >nul
setlocal

rem %~dp0 is the folder this script sits in, so the project can live anywhere.
rem It always ends with a backslash; strip it so the paths below read normally.
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

rem ffmpeg encodes the recording when the meeting ends. Checking for it now
rem rather than then matters: a meeting cannot be re-recorded, and the audio
rem parts would sit unencoded in inbox/ waiting for a tool nobody installed.
where ffmpeg >NUL 2>&1
if errorlevel 1 (
    echo [Error] ffmpeg not found on PATH.
    echo         Install it, then open a NEW terminal so PATH picks it up:
    echo             winget install Gyan.FFmpeg
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

if not exist ".\.venv\Scripts\python.exe" (
    echo [Error] Python venv not found at %PROJECT_DIR%\.venv
    pause
    exit /b 1
)

rem --- Start the watcher in its own window, only if not already running ---
rem Detect by the python command line, not the console window title: on
rem Windows 11 the title belongs to Windows Terminal rather than cmd.exe, so
rem the old tasklist/windowtitle check never matched and every run of this
rem script stacked another watcher (they race for inbox files and the second
rem one dies loading models into a GPU the first already filled).
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*src.main*' }) { exit 0 } exit 1" >NUL 2>&1
if errorlevel 1 (
    echo Starting watcher...
    start "MeetingWatcher" cmd /k "cd /d %PROJECT_DIR% && .\.venv\Scripts\python.exe -m src.main"
    timeout /t 3 /nobreak >nul
) else (
    echo Watcher is already running.
)

rem --- Preflight: a real meeting cannot be re-recorded, so verify the audio
rem --- paths BEFORE it starts, not from the transcript afterwards.
echo.
.\.venv\Scripts\python.exe -m src.preflight
if errorlevel 1 (
    echo.
    choice /c YN /m "ผลตรวจไม่ผ่าน ต้องการอัดต่อไปหรือไม่"
    if errorlevel 2 (
        echo ยกเลิกการอัด
        endlocal
        exit /b 1
    )
)

echo.
echo เลือกโมเดลสรุป:
echo   [1] Opus 5    - แม่นสุด  ($5/$25 ต่อ MTok)
echo   [2] Sonnet 5  - ประหยัด  ($3/$15 ต่อ MTok)
rem Anything that is not "2" lands on Opus 5, so a typo cannot reach Python as an
rem invalid model id. set /p rather than choice: choice takes a single keypress
rem and ignores Enter, and pressing Enter for the default is the point here.
set "MODEL_CHOICE=1"
set /p MODEL_CHOICE=เลือก [1/2] (Enter=1):
if "%MODEL_CHOICE%"=="2" (
    set "MODEL_ID=claude-sonnet-5"
) else (
    set "MODEL_ID=claude-opus-5"
)
echo ใช้โมเดล: %MODEL_ID%

echo.
set /p MEETING_NAME=Meeting name (optional, press Enter to skip):

echo.
echo Recording started. Press Ctrl+C when the meeting ends.
echo.

if "%MEETING_NAME%"=="" (
    .\.venv\Scripts\python.exe -m src.record --model %MODEL_ID%
) else (
    rem "--" ends argparse option parsing, so a meeting name starting with "-"
    rem (e.g. "-standup") is taken as the positional name instead of an option.
    .\.venv\Scripts\python.exe -m src.record --model %MODEL_ID% -- "%MEETING_NAME%"
)

endlocal
pause
