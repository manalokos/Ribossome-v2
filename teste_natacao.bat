@echo off
rem Cenario de teste da natacao (o mesmo que o exemplo probe_jitter):
rem 300 nadadores construidos, terreno plano, sem morte nem fome.
rem RIBO_FSO=1 -> natacao so pelo fluido; RIBO_FSO=0 -> natacao RFT.
cd /d "%~dp0"
set RIBO_LAB=
set RIBO_TERRAIN=plano
set RIBO_SWIMMERS=300
if "%RIBO_FSO%"=="" set RIBO_FSO=1
cargo run --release
if errorlevel 1 pause
