param(
    [string]$MatlabPath = 'D:\Program Files (x86)\MATLAB\R2025a\bin\matlab.exe',
    [ValidateSet('report', 'upstream-script')]
    [string]$ParameterMode = 'upstream-script',
    [ValidateSet('normal', 'accelerator')]
    [string]$SimulationMode = 'accelerator',
    [ValidateSet('report-csv', 'costep-mat')]
    [string]$DataFormat = 'report-csv',
    [string]$DataOutput = '',
    [string]$ScoreOutput = '',
    [switch]$SkipSimulation
)
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if ($ParameterMode -eq 'report') { $modeSuffix = '' } else { $modeSuffix = '_upstream_script' }
if (-not $DataOutput) { $DataOutput = Join-Path $PSScriptRoot "results_costep_data$modeSuffix" }
if (-not $ScoreOutput) {
    $scoreSuffix = $modeSuffix
    if ($DataFormat -eq 'report-csv') {
        if ($ParameterMode -eq 'upstream-script') { $scoreSuffix = '_upstream_csv' } else { $scoreSuffix = '_csv' }
    }
    $ScoreOutput = Join-Path $PSScriptRoot "results_costep_report$scoreSuffix"
}
$DataOutput = [System.IO.Path]::GetFullPath($DataOutput)
$ScoreOutput = [System.IO.Path]::GetFullPath($ScoreOutput)

& $taskPython -X utf8 (Join-Path $PSScriptRoot 'costep_audit.py')
if ($LASTEXITCODE -ne 0) { throw 'Official COSTEP parameter audit failed.' }
if (-not $SkipSimulation) {
    if (-not (Test-Path -LiteralPath $MatlabPath -PathType Leaf)) {
        throw "MATLAB executable does not exist: $MatlabPath"
    }
    $matlabScripts = $PSScriptRoot.Replace("'", "''")
    $matlabData = $DataOutput.Replace("'", "''")
    $matlabSource = (Join-Path $taskRoot 'src\tmp\report-reproduction\costep-source').Replace("'", "''")
    $matlabCommand = "addpath('$matlabScripts'); simulate_costep(SourceDir='$matlabSource',OutputDir='$matlabData',ParameterMode='$ParameterMode',SimulationMode='$SimulationMode'); export_costep_truth('$matlabData');"
    & $MatlabPath -batch $matlabCommand
    if ($LASTEXITCODE -ne 0) { throw 'COSTEP simulation did not complete.' }
}

& $taskPython -X utf8 -u (Join-Path $PSScriptRoot 'run.py') tep `
    --tep-format $DataFormat --tep-root $DataOutput --output $ScoreOutput
if ($LASTEXITCODE -ne 0) { throw 'Detector scoring failed.' }
& $taskPython -X utf8 (Join-Path $PSScriptRoot 'compare_costep.py') `
    --data $DataOutput --scores $ScoreOutput --output (Join-Path $ScoreOutput 'comparison')
if ($LASTEXITCODE -ne 0) { throw 'COSTEP output verification failed.' }
