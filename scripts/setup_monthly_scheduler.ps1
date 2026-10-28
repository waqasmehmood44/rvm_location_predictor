# Windows Task Scheduler registration script for Coripet Monthly Model Retrain
$TaskName = "Coripet_Monthly_Model_Retrain"
$ActionPath = "c:\Users\waqas\Desktop\Coripet_RVM_Predictor\scripts\run_monthly_retrain.bat"
$Time = "05:00"

Write-Host "Registering Windows Scheduled Task: $TaskName to run on the 1st of every month at $Time..." -ForegroundColor Cyan

# Schedule to run on the 1st day of every month at 05:00 AM
schtasks /create /tn "$TaskName" /tr "$ActionPath" /sc monthly /d 1 /st $Time /f

if ($LASTEXITCODE -eq 0) {
    Write-Host "Task '$TaskName' registered successfully!" -ForegroundColor Green
    schtasks /query /tn "$TaskName" /fo LIST
} else {
    Write-Host "Failed to register scheduled task. Ensure you have appropriate permissions." -ForegroundColor Red
}
