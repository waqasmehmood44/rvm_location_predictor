@echo off
REM Coripet RVM Daily Automated ETL Sync Runner
cd /d "c:\Users\waqas\Desktop\Coripet_RVM_Predictor"
echo [%date% %time%] Starting Daily ETL Sync... >> "logs\scheduler_output.log"
"C:\Program Files\Python311\python.exe" "scripts\daily_automated_etl.py" --days 2 >> "logs\scheduler_output.log" 2>&1
echo [%date% %time%] Finished Daily ETL Sync with exit code %errorlevel% >> "logs\scheduler_output.log"
