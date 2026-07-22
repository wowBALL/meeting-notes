@echo off
chcp 65001 >nul
setlocal

set "PROJECT_DIR=D:\COWORK\meeting-notes"
set "FFMPEG1=C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
set "FFMPEG2=C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build-shared\bin"
set "PATH=%FFMPEG1%;%FFMPEG2%;%PATH%"

cd /d "%PROJECT_DIR%"

if not exist ".\.venv\Scripts\python.exe" (
    echo [Error] Python venv not found at %PROJECT_DIR%\.venv
    pause
    exit /b 1
)

rem --- Start the watcher in its own window, only if not already running ---
tasklist /fi "windowtitle eq MeetingWatcher" 2>NUL | find /I "cmd.exe" >NUL
if errorlevel 1 (
    echo Starting watcher...
    start "MeetingWatcher" cmd /k "cd /d %PROJECT_DIR% && set PATH=%FFMPEG1%;%FFMPEG2%;%PATH% && .\.venv\Scripts\python.exe -m src.main"
    timeout /t 3 /nobreak >nul
) else (
    echo Watcher is already running.
)

echo.
set /p MEETING_NAME=Meeting name (optional, press Enter to skip):

echo.
echo Recording started. Press Ctrl+C when the meeting ends.
echo.

if "%MEETING_NAME%"=="" (
    .\.venv\Scripts\python.exe -m src.record
) else (
    .\.venv\Scripts\python.exe -m src.record "%MEETING_NAME%"
)

endlocal
pause
