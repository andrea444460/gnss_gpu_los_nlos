@echo off
cd /d "%~dp0"
pythonw hcn2rinex_gui.py 2>nul || python hcn2rinex_gui.py
