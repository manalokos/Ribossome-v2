@echo off
rem Ribossome v4 em modo laboratorio: piscina 1024^2, sem fluido nem terreno,
rem monomeros ativados por igual e reativacao uniforme.
cd /d "%~dp0"
set RIBO_LAB=1
cargo run --release
if errorlevel 1 pause
