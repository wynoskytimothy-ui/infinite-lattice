# Refresh PATH (needed in terminals opened before Node.js install)
$env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path', 'User')

Set-Location (Split-Path $PSScriptRoot -Parent)

$target = $args[0]
switch ($target) {
  'ch1' { npm run book:ch1 }
  'ch2' { npm run book:ch2 }
  'ch3d' { npm run book:ch3d }
  'ch3' { npm run book:ch3 }
  'ch4' { npm run book:ch4 }
  'ch5' { npm run book:ch5 }
  'ch6' { npm run book:ch6 }
  'front' { npm run book:front }
  'appendices' { npm run book:appendices }
  'all' { npm run book:front; npm run book:ch1; npm run book:ch2; npm run book:ch3d; npm run book:ch3; npm run book:ch4; npm run book:ch5; npm run book:ch6; npm run book:appendices }
  'full' { npm run book:full }
  'pdf' { npm run book:pdf }
  default {
    if (-not (Test-Path 'node_modules')) { npm install }
    Write-Host "Usage: .\book\build.ps1 ch1 | ch2 | front | appendices | all | full | pdf"
  }
}
