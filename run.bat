@echo off
rem Ribossome v4 no mundo completo (2048^2, fluido, terreno, luz).
cd /d "%~dp0"
set RIBO_LAB=
cargo run --release
if errorlevel 1 pause
