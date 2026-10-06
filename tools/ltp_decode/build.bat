@echo off
setlocal

pushd "%~dp0"
for %%I in (
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) do (
  if exist %%~I (
    call "%%~I" >nul
    goto :compile
  )
)
echo MSVC not found. Install: winget install Microsoft.VisualStudio.2022.BuildTools
popd
exit /b 1

:compile
cl /nologo /EHsc ltp_decode_test.cpp /Feltp_decode_test.exe
set ERR=%ERRORLEVEL%
if %ERR% neq 0 goto :done
cl /nologo /EHsc ltp_laser.cpp /Feltp_laser.exe
set ERR=%ERRORLEVEL%
if %ERR% neq 0 goto :done
echo Built ltp_decode_test.exe and ltp_laser.exe in %~dp0

:done
popd
exit /b %ERR%
