@echo off
setlocal
set SRC=%~dp0..\src
set OUT=%~dp0

where cl >nul 2>nul
if errorlevel 1 goto :vcenv
where rc >nul 2>nul
if errorlevel 1 goto :vcenv
goto :build

:vcenv
if not defined VSPATH (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -property installationPath`) do set VSPATH=%%i
)
if not defined VSPATH (echo Visual Studio installation not found ^(vswhere^) & exit /b 1)
if not exist "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" (echo vcvars64.bat not found: "%VSPATH%" & exit /b 1)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (echo vcvars64.bat initialization FAILED & exit /b 1)
where cl >nul 2>nul
if errorlevel 1 (echo cl.exe not found after vcvars64.bat & exit /b 1)
where rc >nul 2>nul
if errorlevel 1 (echo rc.exe not found after vcvars64.bat & exit /b 1)

:build
echo Building shade.exe from src\gpt.cpp ...
rc /nologo /fo"%OUT%shade.res" "%SRC%\shade.rc"
if %ERRORLEVEL% neq 0 (echo shade.res FAILED & exit /b 1)
cl /nologo /std:c++17 /O2 /arch:AVX /openmp /EHsc /I"%SRC%" "%SRC%\gpt.cpp" "%OUT%shade.res" /Fe:"%OUT%shade.exe"
if %ERRORLEVEL% neq 0 (echo shade.exe FAILED & exit /b 1)

echo Done.
