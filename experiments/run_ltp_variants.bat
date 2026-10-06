@echo off
set PYTHONPATH=c:\Users\Andrea\Documents\gnss_gpu\python
set SCRIPT=c:\Users\Andrea\Documents\gnss_gpu\experiments\ltp_to_las.py
set LTP=C:\Users\Andrea\Desktop\2026-03-24-123712_002.ltp
set POS=C:\Users\Andrea\Desktop\lidarcarLCOdo.pos
set CIL=C:\Users\Andrea\Desktop\20260324_111523_S.cil
set IMR=C:\Users\Andrea\Desktop\Rover_GNSS\20260324_111523_I.imr
set COMMON=--ltp %LTP% --pos %POS% --cil %CIL% --max-frames 15000 --stride 4

python %SCRIPT% %COMMON% --out C:\Users\Andrea\Desktop\nuvola_try_base.las
python %SCRIPT% %COMMON% --out C:\Users\Andrea\Desktop\nuvola_try_left.las --scan-side left
python %SCRIPT% %COMMON% --out C:\Users\Andrea\Desktop\nuvola_try_invert.las --invert-elev
python %SCRIPT% %COMMON% --out C:\Users\Andrea\Desktop\nuvola_try_left_invert.las --scan-side left --invert-elev
python %SCRIPT% %COMMON% --out C:\Users\Andrea\Desktop\nuvola_try_imu_pitch.las --imr %IMR% --imu-attitude pitch
python %SCRIPT% %COMMON% --out C:\Users\Andrea\Desktop\nuvola_try_left_imu.las --scan-side left --imr %IMR% --imu-attitude pitch
