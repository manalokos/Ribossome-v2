@echo off
rem Planetas: acrecao num disco 2D (pacotes de massa com inercia numa grelha).
cd /d "%~dp0"
cargo run --release --bin planetas
if errorlevel 1 pause
