# watchdog.ps1 — monitor and auto-restart Windows pipeline scripts
#
# Manages:
#   sh\pipeline\1_motion_diff.ps1  - motion detection from cameras
#   sh\transfer\2_send.ps1         - send files to Linux transfer server
#
# Each script handles its own logging:
#   1_motion_diff: .output\pipeline\1_motion_diff\meta\<date>\run.log
#   2_send:        .output\logs\2_send.log
#   watchdog:      .output\logs\watchdog.log
#
# Usage:
#   .\sh\system\watchdog.ps1             - start watchdog (stays in foreground)
#   .\sh\system\watchdog.ps1 -Register   - register as scheduled task (boot + wake)
#   .\sh\system\watchdog.ps1 -Unregister - remove scheduled task

param(
    [switch] $Register,
    [switch] $Unregister,
    [int]    $CheckSec = 30
)

$ErrorActionPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8

$Repo     = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TaskName = "VideoWatchdog"
$LogDir   = Join-Path $Repo ".output\logs"
$LogFile  = Join-Path $LogDir "watchdog.log"

$Watch = @(
    @{ file = "1_motion_diff.ps1"; path = "$Repo\sh\pipeline\1_motion_diff.ps1" },
    @{ file = "2_send.ps1";        path = "$Repo\sh\transfer\2_send.ps1" }
)

# --- Scheduled Task registration ---

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task '$TaskName' removed."
    exit
}

if ($Register) {
    $ps   = (Get-Command powershell.exe).Source
    $self = $PSCommandPath
    $xml  = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <BootTrigger><Enabled>true</Enabled><Delay>PT10S</Delay></BootTrigger>
    <EventTrigger>
      <Enabled>true</Enabled>
      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
    </EventTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>$ps</Command>
      <Arguments>-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "$self"</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure><Interval>PT1M</Interval><Count>5</Count></RestartOnFailure>
  </Settings>
</Task>
"@
    $xmlPath = "$env:TEMP\watchdog_task.xml"
    $xml | Out-File $xmlPath -Encoding Unicode
    schtasks /Create /TN $TaskName /XML $xmlPath /F | Out-Null
    Remove-Item $xmlPath -ErrorAction SilentlyContinue
    Write-Host "Task '$TaskName' registered (boot + wake from sleep)."
    Write-Host "To remove: .\sh\system\watchdog.ps1 -Unregister"
    exit
}

# --- Helpers ---

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log([string]$msg) {
    $ts   = Get-Date -Format "HH:mm:ss"
    $line = "$ts  $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Find-Running([string]$file) {
    # Returns matching Win32_Process objects; exclude the watchdog itself
    return @(Get-WmiObject Win32_Process |
             Where-Object { $_.CommandLine -like "*$file*" -and
                            $_.CommandLine -notlike "*watchdog*" })
}

function Start-Script([string]$label, [string]$path) {
    Start-Process powershell.exe `
        -ArgumentList "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$path`"" `
        -WindowStyle Hidden
    Start-Sleep 3
    $found = Find-Running (Split-Path $path -Leaf)
    if ($found) {
        $pids = ($found | ForEach-Object { $_.ProcessId }) -join ", "
        Log "  started $label  PID: $pids"
    } else {
        Log "  FAILED to start $label"
    }
}

# --- Main loop ---

Log "=== watchdog start ==="

while ($true) {
    foreach ($s in $Watch) {
        $procs = Find-Running $s.file
        if (-not $procs) {
            Log "not running: $($s.file) - starting..."
            Start-Script $s.file $s.path
        }
    }
    Start-Sleep $CheckSec
}
