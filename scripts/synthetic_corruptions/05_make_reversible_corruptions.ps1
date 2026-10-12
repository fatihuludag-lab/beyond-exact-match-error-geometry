param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [string]$SmilesColumn = "smiles",
    [string]$IdColumn = "sample_id",
    [int]$PerMolecule = 2,
    [int]$MaxInverseTrials = 64
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path data\generated | Out-Null
ssd make-synthetic `
  --input $InputFile `
  --output data\generated\reversible_corruptions.parquet `
  --smiles-col $SmilesColumn `
  --id-col $IdColumn `
  --per-molecule $PerMolecule `
  --seed 20261003 `
  --max-candidates-per-family 5000 `
  --require-reversible `
  --inverse-max-candidates 50000 `
  --max-inverse-trials $MaxInverseTrials

Write-Host "data\generated\reversible_corruptions.parquet olusturuldu." -ForegroundColor Green
