@echo off
cd /d "D:\Users\ShadeFurryIT\Desktop\ShadeAI C++ Test\build"
start "shade-train-128" shade_omp.exe train gpt --data ../data/train_full3.txt --epochs 3 --ctx 128 --dim 256 --blocks 8 --window 100000 --continue --setpin --log ../data/train_log
