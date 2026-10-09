@echo off
rem Ribossome: microscopio 3D (prototipo). Abre o autosave, ou a cena em SCENE.
rem Rato: arrastar roda a camara, botao direito desloca, roda aproxima.
rem Espaco pausa a simulacao (a imagem converge); F/G fecham e abrem o diafragma.
cd /d "%~dp0"
cargo run --release --bin microscopio
if errorlevel 1 pause
