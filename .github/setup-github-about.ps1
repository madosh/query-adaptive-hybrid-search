# Apply GitHub About description + topics (requires GitHub CLI: gh auth login)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$Description = (Get-Content "$PSScriptRoot\DESCRIPTION.txt" -Raw).Trim()
$Topics = (Get-Content "$PSScriptRoot\TOPICS.txt" | Where-Object { $_.Trim() -ne "" }) -join ","

Write-Host "Setting repository description..."
gh repo edit --description $Description

Write-Host "Adding repository topics..."
gh repo edit --add-topic $Topics

Write-Host "Done. Verify in GitHub -> Settings -> General -> About."
