# Started by start.bat. The server lives in a Windows job that is destroyed
# when this console exits, so closing the bat window stops Python too.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$Port = 8765

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WinJob {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);
    [DllImport("kernel32.dll")]
    public static extern bool SetInformationJobObject(IntPtr hJob, int infoClass, IntPtr info, uint length);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
}
"@

function Find-Python {
    $cmds = @(Get-Command python -All -ErrorAction SilentlyContinue)
    foreach ($c in $cmds) {
        if ($c.Source -and $c.Source -notmatch "WindowsApps") { return $c.Source }
    }
    $fallback = Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"
    if (Test-Path $fallback) { return $fallback }
    throw "Python not found. Install Python 3 and add it to PATH."
}

function Stop-PortListener([int]$port) {
    $lines = netstat -ano | Select-String ":$port\s+.*LISTENING"
    foreach ($line in $lines) {
        $procId = ($line.ToString() -split "\s+")[-1]
        if ($procId -match "^\d+$" -and [int]$procId -gt 0) {
            Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue
        }
    }
}

Stop-PortListener $Port
Start-Sleep -Milliseconds 300

$job = [WinJob]::CreateJobObject([IntPtr]::Zero, $null)
if ($job -eq [IntPtr]::Zero) { throw "CreateJobObject failed" }
# JOBOBJECT_EXTENDED_LIMIT_INFORMATION, x64. LimitFlags at offset 16.
# 0x2000 = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
$infoSize = 144
$info = [Runtime.InteropServices.Marshal]::AllocHGlobal($infoSize)
try {
    [Runtime.InteropServices.Marshal]::Copy([byte[]]::new($infoSize), 0, $info, $infoSize)
    [Runtime.InteropServices.Marshal]::WriteInt32($info, 16, 0x2000)
    $ok = [WinJob]::SetInformationJobObject($job, 9, $info, [uint32]$infoSize)
    if (-not $ok) { throw "SetInformationJobObject failed" }
} finally {
    [Runtime.InteropServices.Marshal]::FreeHGlobal($info)
}

$python = Find-Python
$proc = Start-Process -FilePath $python -ArgumentList "-u", "serve.py" -WorkingDirectory $root -PassThru -NoNewWindow
if (-not [WinJob]::AssignProcessToJobObject($job, $proc.Handle)) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Could not tie the server to this window."
}

Start-Sleep -Milliseconds 400
Start-Process "http://127.0.0.1:$Port/"
Wait-Process -Id $proc.Id
