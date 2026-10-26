@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto failed

echo LostFleet Tools - Windows local launcher
echo.
if not exist "lostfleet_tools\server.py" goto incomplete
if not exist "requirements.txt" goto incomplete

if exist ".venv\Scripts\python.exe" goto environment_ready

py -3.12 -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto try_python
py -3.12 -m venv .venv
if errorlevel 1 goto failed
goto environment_ready

:try_python
python -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto missing_python
python -m venv .venv
if errorlevel 1 goto failed

:environment_ready
".venv\Scripts\python.exe" -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8"
if errorlevel 1 goto wrong_environment
echo Installing or checking project dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-input -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import sys,PIL,cv2,numpy; print('Python',sys.version.split()[0]); print('Pillow',PIL.__version__,'OpenCV',cv2.__version__,'NumPy',numpy.__version__)"
if errorlevel 1 goto failed

echo.
echo Starting http://127.0.0.1:8000 in your default browser.
echo Keep this window open. Press Ctrl+C to stop the server.
echo If the browser does not open, enter the address manually.
echo.
".venv\Scripts\python.exe" -m lostfleet_tools.server --host 127.0.0.1 --port 8000 --open-browser
if errorlevel 1 goto failed
exit /b 0

:missing_python
echo ERROR: Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/
echo Enable the Python launcher or Add python.exe to PATH, then run this file again.
pause
exit /b 1

:wrong_environment
echo ERROR: The existing .venv does not use 64-bit Python 3.12.
echo Preserve or rename it, then run this launcher to create a new .venv.
pause
exit /b 1

:incomplete
echo ERROR: Project files are missing. Extract the complete project ZIP first.
echo Do not run this file from inside the ZIP viewer.
pause
exit /b 1

:failed
echo.
echo ERROR: Startup failed. Read the message above.
echo Check your Python installation, package download connection, and port 8000.
echo See README.md for manual startup and troubleshooting.
pause
exit /b 1
