@echo off
pushd "%~dp0"
for %%I in (
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) do if exist %%~I (call "%%~I" >nul & goto :go)
echo MSVC not found
exit /b 1
:go
cl /nologo /EHsc ltp_laser.cpp /Feltp_laser.exe
exit /b %ERRORLEVEL%
