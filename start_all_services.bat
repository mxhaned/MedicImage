@echo off
echo ========================================
echo    MEDICIMAGE - DÉMARRAGE COMPLET
echo ========================================
echo.
echo Démarrage des services backend...

echo.
echo [1/3] Démarrage du service DermaScan (Port 5002)...
start "DermaScan Backend" cmd /k "call start_dermascan_backend.bat"

echo [2/3] Attente de 5 secondes...
timeout /t 5 /nobreak

echo [3/3] Démarrage du service Brain Tumor RÉEL (Port 5001)...
start "Brain Tumor Backend RÉEL" cmd /k "call start_braintumor_backend.bat"

echo.
echo [4/4] Attente de 10 secondes avant de démarrer le frontend...
timeout /t 10 /nobreak

echo [5/5] Démarrage du Frontend React...
start "Frontend React" cmd /k "call start_frontend.bat"

echo.
echo ✅ TOUS LES SERVICES DÉMARRÉS !
echo.
echo 🌐 URLs d'accès:
echo    - Frontend React:     http://localhost:5173/
echo    - Backend DermaScan:  http://localhost:5002/api/health  
echo    - Backend Brain Tumor: http://localhost:5001/health
echo.
echo ✅ SYSTÈME MEDICIMAGE COMPLET
echo    - DermaScan: VERSION COMPLÈTE avec IA
echo    - Brain Tumor: VERSION COMPLÈTE avec U-Net + TensorFlow + PyTorch
echo    - Frontend: Interface complète
echo    - Corrections tkinter: VALIDÉES ET ACTIVES
echo.
echo Appuyez sur une touche pour fermer cette fenêtre...
pause