@echo off
chcp 65001 >nul
setlocal

rem ทางเข้าแบบหน้าจอ -- อยู่คู่กับ start-meeting.bat ไม่ได้แทนที่มัน ทั้งสองทาง
rem จบที่ inbox/ เหมือนกัน ตัวประมวลผลจึงไม่ต้องรู้ว่าประชุมมาจากทางไหน
rem
rem อย่าเปิดสองทางพร้อมกัน: ไม่มีล็อกกันไว้ ทั้งคู่จะอัดพร้อมกันได้จริงและ
rem ได้ไฟล์แยกกันสองไฟล์ ซึ่งเสียแรงกับค่า API เปล่า ๆ

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

rem ffmpeg ใช้ตอนจบประชุม เช็คตอนนี้ไม่ใช่ตอนนั้น: ประชุมอัดซ้ำไม่ได้ และชิ้นส่วน
rem เสียงจะค้างอยู่ใน inbox/ รอเครื่องมือที่ไม่มีใครติดตั้ง
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

rem อ่านพอร์ตจาก .env โดยไม่ผ่าน load_config เพราะ load_config บังคับต้องมี
rem ANTHROPIC_API_KEY/HF_TOKEN ซึ่งจะทำให้บรรทัดนี้ล้มด้วยเหตุผลที่ไม่เกี่ยวกัน
set "RUNNER_PORT=8765"
for /f "usebackq delims=" %%p in (`.\.venv\Scripts\python.exe -c "import os;from dotenv import load_dotenv;load_dotenv('.env');print(os.environ.get('UI_PORT','8765'))"`) do set "RUNNER_PORT=%%p"

rem ตรวจด้วย command line ของ python ไม่ใช่ชื่อหน้าต่าง: บน Windows 11 ชื่อ
rem หน้าต่างเป็นของ Windows Terminal ไม่ใช่ cmd.exe การเช็คด้วยชื่อจึงไม่เคยตรง
rem แล้วทุกครั้งที่รันจะซ้อน watcher เพิ่มอีกตัว (แย่งไฟล์ใน inbox กัน และตัวที่
rem สองตายตอนโหลดโมเดลลง GPU ที่ตัวแรกจองไปแล้ว)
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*src.main*' }) { exit 0 } exit 1" >NUL 2>&1
if errorlevel 1 (
    echo Starting watcher...
    start "MeetingWatcher" cmd /k "cd /d %PROJECT_DIR% && .\.venv\Scripts\python.exe -m src.main"
    timeout /t 3 /nobreak >nul
) else (
    echo Watcher is already running.
)

powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*src.session_service*' }) { exit 0 } exit 1" >NUL 2>&1
if errorlevel 1 (
    echo Starting the runner UI on port %RUNNER_PORT% ...
    start "MeetingRunnerUI" cmd /k "cd /d %PROJECT_DIR% && .\.venv\Scripts\python.exe -m src.session_service"
    timeout /t 3 /nobreak >nul
) else (
    echo Runner UI is already running.
)

start "" http://127.0.0.1:%RUNNER_PORT%

endlocal
