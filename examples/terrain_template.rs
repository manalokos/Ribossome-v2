//! Grava o terreno gerado por omissão (com a fumarola) em PNG, como modelo
//! para editar: azul = terreno, vermelho = calor por píxel. OUT = caminho.
use ribossome::params::{Fumarole, WorldConfig};
use ribossome::world::terrain;

fn main() {
    let cfg = WorldConfig::DEFAULT;
    let fum = vec![Fumarole::v3_default()];
    let g = terrain::generate(&cfg, 1, &fum);
    let out = std::env::var("OUT").unwrap_or_else(|_| "terreno.png".into());
    terrain::save_png(std::path::Path::new(&out), &cfg, &g, &terrain::rasterize_fumaroles(&cfg, &fum)).unwrap();
    println!("{out}");
}
