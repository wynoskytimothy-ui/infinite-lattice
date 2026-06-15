# Stop AETHOS benchmark Python jobs (eval_beir, run_ab, run_aethos) to reduce CPU load.
$pattern = 'eval_beir|run_ab|run_aethos'
$stopped = 0
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and ($_.CommandLine -match $pattern) } |
    ForEach-Object {
        Write-Host "Stopping PID $($_.ProcessId): $($_.CommandLine.Substring(0, [Math]::Min(80, $_.CommandLine.Length)))..."
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $stopped++
    }
if ($stopped -eq 0) {
    Write-Host "No eval_beir/run_ab python processes found."
} else {
    Write-Host "Stopped $stopped process(es). CPU should drop shortly."
}
