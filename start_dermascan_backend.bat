@echo off
echo Starting MedicImage DermaScan Backend (FastAPI)...
echo.

cd Backend

echo Activating virtual environment...
cd ..
call venv\Scripts\activate.bat

echo Starting DermaScan FastAPI server...
cd Backend
python app.py

pause