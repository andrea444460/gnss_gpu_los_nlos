@echo off
for %%I in (
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do if exist %%~I (call "%%~I" >nul & goto :go)
exit /b 1
:go
dumpbin /exports "C:\Program Files\CHCNAV\CoPre2\Release\LTLaser.dll" > "%~dp0exports_lt_laser.txt" 2>&1
findstr /i "EP Calib init open parse set load param" "%~dp0exports_lt_laser.txt"
