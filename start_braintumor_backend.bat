@echo off
echo ========================================
echo    BRAIN TUMOR BACKEND - DÉMARRAGE
echo ========================================
echo.

cd Backend

echo 🔧 Configuration anti-tkinter...
set MPLBACKEND=Agg
set PYTHONWARNINGS=ignore::UserWarning:matplotlib

echo Activating virtual environment...
cd ..
call venv\Scripts\activate.bat

echo.
echo ✅ Corrections tkinter appliquées:
echo    - Backend matplotlib non-interactif
echo    - Nettoyage automatique des ressources  
echo    - Warnings tkinter désactivés
echo.
echo Starting Brain Tumor Flask server...
echo URL d'accès: http://localhost:5001/
echo Endpoint de test: http://localhost:5001/health
echo.

cd Backend
python app2.py

pause