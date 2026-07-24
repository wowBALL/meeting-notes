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
rem Detect by the python command line, not the console window title: on
rem Windows 11 the title belongs to Windows Terminal rather than cmd.exe, so
rem the old tasklist/windowtitle check never matched and every run of this
rem script stacked another watcher (they race for inbox files and the second
rem one dies loading models into a GPU the first already filled).
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*src.main*' }) { exit 0 } exit 1" >NUL 2>&1
if errorlevel 1 (
    echo Starting watcher...
    start "MeetingWatcher" cmd /k "cd /d %PROJECT_DIR% && set PATH=%FFMPEG1%;%FFMPEG2%;%PATH% && .\.venv\Scripts\python.exe -m src.main"
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
