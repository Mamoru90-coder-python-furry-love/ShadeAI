@echo off
set PATH=C:\msys64\ucrt64\bin;%PATH%
set CXX=C:\msys64\ucrt64\bin\g++.exe
set FLAGS=-std=c++17 -O2 -Wall -march=native -msse4.2 -static-libgcc -static-libstdc++ -static -DNDEBUG -fopenmp
set SRC=D:\Users\ShadeFurryIT\Desktop\ShadeAI C++ Test\src
set OUT=D:\Users\ShadeFurryIT\Desktop\ShadeAI C++ Test\build

echo Building shade.exe ...
"%CXX%" %FLAGS% -I"%SRC%" "%SRC%\shade.cpp" -lws2_32 -lshell32 -o "%OUT%\shade.exe"
if %ERRORLEVEL% neq 0 (echo shade.exe FAILED & exit /b 1)

copy /y "%OUT%\shade.exe" "%OUT%\shade_omp.exe" >nul
echo Done.
