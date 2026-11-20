@echo off
setlocal
set SRC=%~dp0..\src
set OUT=%~dp0

where cl >nul 2>nul
if not errorlevel 1 goto :build
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"

:build
echo Building shade.exe from src\gpt.cpp ...
cl /nologo /std:c++17 /O2 /arch:AVX /openmp /EHsc /I"%SRC%" "%SRC%\gpt.cpp" /Fe:"%OUT%shade.exe"
if %ERRORLEVEL% neq 0 (echo shade.exe FAILED & exit /b 1)

echo Done.
