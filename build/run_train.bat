@echo off
setlocal
set EXE=%~dp0shade.exe
set WD=%~dp0
set LOG=%WD%train_run.log
set ERR=%WD%train_run.err
del /q "%LOG%" "%ERR%" 2>nul

start "" /b /high "%EXE%" train chat --data ../data/dialogues.jsonl --epochs 3 --ctx 2048 --dim 256 --blocks 8 --lr 0.0001 --batch 16 --continue --dashboard --port 8765 > "%LOG%" 2> "%ERR%"
echo launched at %TIME%
