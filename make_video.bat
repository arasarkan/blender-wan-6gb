@echo off
rem Drag a .blend file onto this file, or double-click it to try the built-in test scene.
rem From a terminal: make_video.bat scene.blend --prompt "..." --seconds 5 --fps 24
setlocal
cd /d "%~dp0"

if not exist config.json (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set "PY="
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "(Get-Content -Raw config.json | ConvertFrom-Json).python"`) do set "PY=%%p"
if not defined PY (
  echo config.json has no python entry; run setup.bat again.
  pause
  exit /b 1
)

"%PY%" "%~dp0make_video.py" %*
set "RC=%ERRORLEVEL%"

rem keep the window open after a double-click or drag and drop (no options given)
if "%~2"=="" (echo. & pause)
exit /b %RC%
