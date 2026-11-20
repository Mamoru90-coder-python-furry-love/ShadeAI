@echo off
cd /d "D:\Users\ShadeFurryIT\Desktop\ShadeAI C++ Test\build"
call compile.bat
if %ERRORLEVEL% neq 0 (echo BUILD FAILED & pause & exit /b 1)
start "" shade.exe train gpt --data ..\data\ukr_fiction.txt --epochs 1 --window 12000 --ctx 64 --dim 256 --blocks 8 --continue --dashboard --log ..\data\dash_test
echo.
echo Дашборд відкриється у браузері: http://localhost:8765
echo Розверни обидва вікна поруч і дивись, що твориться під капотом!
pause