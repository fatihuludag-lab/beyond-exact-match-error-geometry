param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [string]$SmilesColumn = "smiles",
    [string]$IdColumn = "sample_id",
    [int]$PerMolecule = 1
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path data\generated | Out-Null
ssd make-synthetic `
  --input $InputFile `
  --output data\generated\controlled_corruptions.parquet `
  --smiles-col $SmilesColumn `
  --id-col $IdColumn `
  --per-molecule $PerMolecule `
  --seed 20261003

Write-Host "data\generated\controlled_corruptions.parquet olusturuldu." -ForegroundColor Green
