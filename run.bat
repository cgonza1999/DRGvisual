@echo off
echo Setting up virtual environment and installing dependencies...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install required packages
pip install -r requirements.txt
py main.py
echo Setup complete! Virtual environment is ready and packages are installed.
pause
