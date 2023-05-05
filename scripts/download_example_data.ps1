cd "$(Split-Path $MyInvocation.MyCommand.Path)/.."

Remove-Item -Recurse data
New-Item -ItemType Directory -Path data

Write-Host "Downloading example data:"
Invoke-WebRequest -Uri "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip" -OutFile "nerf_example_data.zip" -UseBasicParsing
if ($LastExitCode -ne 0) {
    Write-Host "Download failed!"
    exit 1
}

Set-Location data
Write-Host -NoNewline "Unzipping example data..."
Expand-Archive ../nerf_example_data.zip
if ($LastExitCode -ne 0) {
    Write-Host "Unzip failed! Please check if 'Expand-Archive' is installed!"
    exit 1
}

Write-Host

Get-ChildItem -Directory | ForEach-Object {
    Move-Item "$($_.FullName)\*" ./
    Remove-Item $_.FullName
}

Remove-Item ../nerf_example_data.zip
Write-Host "Done."