Set WshShell = CreateObject("WScript.Shell")
Set args = WScript.Arguments
' i keep closing vscode so this is better to run oop

If args.Count > 0 Then
    cmd = Chr(34) & "fulldockerPipeline.bat" & Chr(34) & " " & Chr(34) & args(0) & Chr(34)
Else
    cmd = Chr(34) & "fulldockerPipeline.bat" & Chr(34)
End If

' Run silently
WshShell.Run cmd, 1
WScript.Echo "Process for " & args(0) & " started."

