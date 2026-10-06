@echo off
set "EXE=%~dp0ltp_decode_test.exe"
set "LTP=%~1"
if "%LTP%"=="" set "LTP=C:\Users\Andrea\Desktop\2026-03-24-123712_002.ltp"
if not exist "%EXE%" (
  call "%~dp0build.bat"
)
"%EXE%" "%LTP%" 50
exit /b %ERRORLEVEL%
