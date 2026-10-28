# Windows Task Scheduler registration script for Coripet Daily ETL Sync
$TaskName = "Coripet_Daily_ETL_Sync"
$ActionPath = "c:\Users\waqas\Desktop\Coripet_RVM_Predictor\scripts\run_daily_sync.bat"
$Time = "04:00"

Write-Host "Registering Windows Scheduled Task: $TaskName to run daily at $Time..." -ForegroundColor Cyan

# Use schtasks for native Windows Task Scheduler registration
schtasks /create /tn "$TaskName" /tr "$ActionPath" /sc daily /st $Time /f

if ($LASTEXITCODE -eq 0) {
    Write-Host "Task '$TaskName' registered successfully!" -ForegroundColor Green
    schtasks /query /tn "$TaskName" /fo LIST
} else {
    Write-Host "Failed to register scheduled task. Ensure you have appropriate permissions." -ForegroundColor Red
}
