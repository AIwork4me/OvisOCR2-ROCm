# One-command smoke demo for OvisOCR2-ROCm.
# Runs the no-GPU `smoke` backend against examples/demo.png.
$ErrorActionPreference = "Stop"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO = Split-Path -Parent $HERE
$TMP = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path "$TMP/imgs" | Out-Null
Copy-Item "$HERE/demo.png" "$TMP/imgs/"
python "$REPO/adapter/run_adapter.py" --img-dir "$TMP/imgs" --out-dir "$TMP/out" `
    --platform windows-hip --backend smoke
Write-Host "--- demo output ---"
Get-Content "$TMP/out/*.md"
