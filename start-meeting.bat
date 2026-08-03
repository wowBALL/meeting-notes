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
rem Also require the venv's own python.exe: the WindowsApps python.exe alias
rem matches "python" and *src.main* too but lacks soundfile, so it exits
rem right after failing to import it. Matching on name/cmdline alone made
rem this script see that dead process and skip starting the real watcher,
rem so the real one silently never ran (found 2026-08-03, see memory).
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -like '*src.main*' -and $_.ExecutablePath -like '*\.venv\Scripts\python.exe' }) { exit 0 } exit 1" >NUL 2>&1
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
echo   [1] Qwen 3.6  - ข้อมูลไม่ออกนอกบริษัท (เร็วที่สุด)
echo   [2] GLM 5.2   - ข้อมูลไม่ออกนอกบริษัท (ช้ากว่า)
echo   [3] Opus 5    - แม่นสุด  ($5/$25 ต่อ MTok)
echo   [4] Sonnet 5  - ประหยัด  ($3/$15 ต่อ MTok)
echo   [5] ถอดเสียงอย่างเดียว - ไม่สรุป (ไม่เสียเงิน)
rem Anything that is not "2".."5" lands on Qwen 3.6, so a typo cannot reach
rem Python as an invalid model id. set /p rather than choice: choice takes a single
rem keypress and ignores Enter, and pressing Enter for the default is the point here.
rem Qwen is the default because the transcript stays on company infrastructure with
rem it -- the private path has to be the one that needs no decision -- and of the two
rem in-house options it is the one that actually finishes (see src/llm.py: GLM-5.2
rem spent its whole budget on reasoning and returned empty content 10 times out of 10).
rem
rem The numbering matches web/app.js top to bottom, so the two entry points read the
rem same. That moved transcript-only from [4] to [5] on purpose -- a "4" typed from
rem muscle memory now records with Sonnet 5 instead of skipping the summary.
set "MODEL_CHOICE=1"
set /p MODEL_CHOICE=เลือก [1/2/3/4/5] (Enter=1):
if "%MODEL_CHOICE%"=="2" (
    rem Must stay byte-identical to the registry key in src/llm.py -- the endpoint
    rem rejects a lowercased id.
    set "MODEL_ID=GLM-5.2"
) else if "%MODEL_CHOICE%"=="3" (
    set "MODEL_ID=claude-opus-5"
) else if "%MODEL_CHOICE%"=="4" (
    set "MODEL_ID=claude-sonnet-5"
) else if "%MODEL_CHOICE%"=="5" (
    rem Must stay byte-identical to job.NO_SUMMARY_MODEL: the pipeline decides
    rem whether to skip summarizing by comparing against this exact string.
    set "MODEL_ID=transcript-only"
) else (
    rem Must stay byte-identical to the registry key in src/llm.py, "Qwen/" and
    rem capitalisation included -- the endpoint rejects anything else.
    set "MODEL_ID=Qwen/Qwen3.6-35B-A3B"
)
rem "ใช้โมเดล: transcript-only" would name a model that does not exist, so the
rem confirmation line follows the mode rather than the value.
if "%MODEL_ID%"=="transcript-only" (
    echo โหมด: ถอดเสียงอย่างเดียว ไม่ส่งสรุป
) else (
    echo ใช้โมเดล: %MODEL_ID%
)

rem Named PROFILE_ID, not MEETING_PROFILE: "set" here would also export the name
rem to the child python process, where MEETING_PROFILE is the .env variable and
rem would silently shadow the file the user actually configured.
rem
rem Deliberately NOT wrapped in an "if (...)" block: this script runs under plain
rem setlocal, so %VAR% inside a parenthesized block expands when the block is
rem parsed -- before set /p has run -- and the choice would always read as empty.
rem The fix is a label rather than "setlocal EnableDelayedExpansion", because
rem delayed expansion would eat "!" out of the meeting name prompted for below.
set "PROFILE_ID=dev"
rem The profile only selects which summarization rules apply, and transcript-only
rem produces no summary -- asking would be a question whose answer changes nothing.
if "%MODEL_ID%"=="transcript-only" goto :profile_done

echo.
echo ประเภทประชุม:
echo   [1] dev ล้วน        - ศัพท์เทคนิคตรงๆ
echo   [2] Business + dev  - แยก "ทำได้" ออกจาก "จะทำ" ขยายศัพท์ให้คนนอกทีม
rem Anything that is not "2" lands on dev, so a typo cannot reach Python as an
rem unknown profile. dev is the default because it is three of every four meetings
rem -- and picking cross by mistake is not merely wasteful: the prompt would tell
rem the model that words like "เสร็จ" mean different things to two sides when only
rem dev is in the room, and it starts hedging ordinary statements.
set "PROFILE_CHOICE=1"
set /p PROFILE_CHOICE=เลือก [1/2] (Enter=1):
rem Must stay byte-identical to prompts.CROSS_TEAM_PROFILE and to the filename
rem prompts/profiles/<value>.md
if "%PROFILE_CHOICE%"=="2" set "PROFILE_ID=cross"
echo ใช้ประเภท: %PROFILE_ID%

:profile_done

echo.
echo เลือกตัวถอดเสียง:
echo   [1] Whisper large-v3  - ค่าเริ่มต้น แม่นสุด
echo   [2] Typhoon (ทดลอง)   - เร็วกว่า 3-10 เท่า แต่พลาดศัพท์เฉพาะทาง/glossary ง่ายกว่า
rem Anything that is not "2" lands on whisper, so a typo cannot reach Python as an
rem unknown engine. whisper is the default because it is the proven daily-use path;
rem Typhoon (src/transcribe_typhoon.py) is experimental and needs .typhoon_venv present.
set "ASR_ENGINE_ID=whisper"
set /p ASR_CHOICE=เลือก [1/2] (Enter=1):
if "%ASR_CHOICE%"=="2" set "ASR_ENGINE_ID=typhoon"
echo ใช้ตัวถอดเสียง: %ASR_ENGINE_ID%

echo.
set /p MEETING_NAME=Meeting name (optional, press Enter to skip):

echo.
echo Recording started. Press Ctrl+C when the meeting ends.
echo.

if "%MEETING_NAME%"=="" (
    .\.venv\Scripts\python.exe -m src.record --model %MODEL_ID% --profile %PROFILE_ID% --asr-engine %ASR_ENGINE_ID%
) else (
    rem "--" ends argparse option parsing, so a meeting name starting with "-"
    rem (e.g. "-standup") is taken as the positional name instead of an option.
    .\.venv\Scripts\python.exe -m src.record --model %MODEL_ID% --profile %PROFILE_ID% --asr-engine %ASR_ENGINE_ID% -- "%MEETING_NAME%"
)

endlocal
pause
