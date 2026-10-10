//! O ESTÚDIO do inspetor: fotografa a outra fita dos primeiros agentes vivos
//! de uma cena e grava os retratos (saves/sprites/_estudio_N.png). SCENE (por
//! omissão o autosave), N (quantos, por omissão 3).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::species::reverse_complement;
use ribossome::tree_view::Studio;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let genomes = ribossome::species::living_genomes(&gpu, &w);
    // (Como na aplicação: quando o estúdio é montado já há uma vista, e o
    // atlas de sprites já está na placa.)
    let clock = std::time::Instant::now();
    let _view = ribossome::render::capture::Capture::new(&gpu, &w, 256);
    println!("primeira vista (carrega o atlas) em {:.2} s", clock.elapsed().as_secs_f32());
    let clock = std::time::Instant::now();
    let mut studio = Studio::new(&gpu, &w);
    println!("estúdio montado em {:.2} s", clock.elapsed().as_secs_f32());
    for (i, g) in genomes.iter().filter(|g| g.len() > 60).take(n).enumerate() {
        let clock = std::time::Instant::now();
        match studio.portrait(&gpu, &w, &reverse_complement(g)) {
            Some((rgba, side)) => {
                let file = format!("saves/sprites/_estudio_{i}.png");
                let mut e = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&file).unwrap()), side, rgba.len() as u32 / 4 / side);
                e.set_color(png::ColorType::Rgba);
                e.set_depth(png::BitDepth::Eight);
                e.write_header().unwrap().write_image_data(&rgba).unwrap();
                println!("{file}: {} bases, {:.0} ms", g.len(), clock.elapsed().as_secs_f32() * 1000.0);
            }
            None => println!("genoma {i} ({} bases): não nasceu", g.len()),
        }
    }
}
