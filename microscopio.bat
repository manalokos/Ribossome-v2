@echo off
rem Ribossome: microscopio 3D (prototipo). Abre o autosave, ou a cena em SCENE.
rem Rato: arrastar roda a camara, botao direito desloca, roda aproxima.
rem Comeca parado (a imagem converge); Espaco poe a correr. F/G diafragma,
rem C cor, M monomeros.
cd /d "%~dp0"
cargo run --release --bin microscopio
if errorlevel 1 pause
