param([int]$Workers = 1)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1

ssd audit `
  --input data\generated\controlled_corruptions.parquet `
  --output-dir results\ssd2a_controlled `
  --workers $Workers `
  --max-candidates-per-family 50000

Get-Content results\ssd2a_controlled\summary.json
