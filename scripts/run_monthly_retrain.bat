@echo off
REM ==============================================================================
REM Coripet RVM Predictor - Monthly Automated Model Retraining Launcher
REM Scheduled to run on the 1st of every month at 05:00 AM
REM ==============================================================================

cd /d "c:\Users\waqas\Desktop\Coripet_RVM_Predictor"

set "PYTHON_EXE=C:\Program Files\Python311\python.exe"
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=python"
)

echo [%date% %time%] Launching Monthly Retraining Pipeline... >> "logs\monthly_retrain.log"
"%PYTHON_EXE%" "scripts\monthly_automated_retrain.py" >> "logs\monthly_retrain.log" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] Monthly Retraining Pipeline Finished with exit code: %EXIT_CODE% >> "logs\monthly_retrain.log"
exit /b %EXIT_CODE%
