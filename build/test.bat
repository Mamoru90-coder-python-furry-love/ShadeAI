@echo off
setlocal
set SRC=%~dp0..\src
set OUT=%~dp0
set TESTS=%~dp0..\tests

where cl >nul 2>nul
if not errorlevel 1 goto :build
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -products * -latest -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"

:build
set CFLAGS=/nologo /std:c++17 /O2 /arch:AVX /openmp /EHsc /I"%SRC%"

cl %CFLAGS% "%TESTS%\simd_check.cpp" /Fe:"%OUT%simd_check.exe" || goto :fail
cl %CFLAGS% "%TESTS%\adamw_checks.cpp" /Fe:"%OUT%adamw_checks.exe" || goto :fail
cl %CFLAGS% "%TESTS%\adamw_edge_checks.cpp" /Fe:"%OUT%adamw_edge_checks.exe" || goto :fail
cl /nologo /std:c++17 /O2 /EHsc /I"%SRC%" "%TESTS%\adamw_contract_checks.cpp" /Fe:"%OUT%adamw_contract_checks.exe" || goto :fail
cl /nologo /std:c++17 /O2 /EHsc /I"%SRC%" "%TESTS%\optimizer_resume_checks.cpp" /Fe:"%OUT%optimizer_resume_checks.exe" || goto :fail
cl %CFLAGS% "%TESTS%\gradcheck.cpp" /Fe:"%OUT%gradcheck.exe" || goto :fail
cl /nologo /std:c++17 /O2 /arch:AVX /EHsc /I"%SRC%" "%TESTS%\serialize_checks.cpp" /Fe:"%OUT%serialize_checks.exe" || goto :fail
cl %CFLAGS% "%TESTS%\tokenizer_checks.cpp" /Fe:"%OUT%tokenizer_checks.exe" || goto :fail
cl %CFLAGS% "%TESTS%\gpt_checkpoint_check.cpp" /Fe:"%OUT%gpt_checkpoint_check.exe" || goto :fail
cl %CFLAGS% "%TESTS%\gpt_sample_check.cpp" /Fe:"%OUT%gpt_sample_check.exe" || goto :fail
cl %CFLAGS% "%TESTS%\gpt_resume_checks.cpp" /Fe:"%OUT%gpt_resume_checks.exe" || goto :fail
cl %CFLAGS% "%TESTS%\checkpoint_integrity_check.cpp" /Fe:"%OUT%checkpoint_integrity_check.exe" || goto :fail
cl %CFLAGS% "%TESTS%\memtrack_check.cpp" /Fe:"%OUT%memtrack_check.exe" || goto :fail
cl /nologo /std:c++17 /O2 /arch:AVX /DNDEBUG /EHsc /I"%SRC%" "%TESTS%\tensor_checks.cpp" /Fe:"%OUT%tensor_checks.exe" || goto :fail
cl %CFLAGS% "%SRC%\main.cpp" /Fe:"%OUT%demos.exe" || goto :fail

"%OUT%simd_check.exe" || goto :fail
"%OUT%adamw_checks.exe" || goto :fail
"%OUT%adamw_edge_checks.exe" || goto :fail
"%OUT%adamw_contract_checks.exe" || goto :fail
"%OUT%optimizer_resume_checks.exe" || goto :fail
"%OUT%gradcheck.exe" || goto :fail
"%OUT%serialize_checks.exe" || goto :fail
"%OUT%tokenizer_checks.exe" || goto :fail
"%OUT%gpt_checkpoint_check.exe" || goto :fail
"%OUT%gpt_sample_check.exe" || goto :fail
"%OUT%gpt_resume_checks.exe" || goto :fail
"%OUT%checkpoint_integrity_check.exe" || goto :fail
"%OUT%memtrack_check.exe" || goto :fail
"%OUT%tensor_checks.exe" || goto :fail
"%OUT%demos.exe" >nul || goto :fail

echo All tests passed.
exit /b 0

:fail
echo Tests FAILED.
exit /b 1
