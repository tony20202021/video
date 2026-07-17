Dim fso, scriptDir, watchdogPath, WS
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
watchdogPath = scriptDir & "\watchdog.ps1"
Set WS = CreateObject("WScript.Shell")
WS.Run "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & watchdogPath & """", 0, False
