@echo off
REM Setup script for Cisco Business Aggregator on Windows

echo Setting up Cisco Business Aggregator...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating necessary directories...
python setup_directories.py

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo Please edit the .env file with your API keys and settings.
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit the .env file with your API keys and settings
echo 2. Run the application with: python main.py
echo.
echo To activate the virtual environment in the future, run:
echo venv\Scripts\activate.bat

pause
