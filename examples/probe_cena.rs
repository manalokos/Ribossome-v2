//! Resumo de uma cena gravada, sem correr passos: epoch, parâmetros que
//! diferem dos valores por omissão, média das últimas amostras das
//! estatísticas guardadas na cena e duas imagens (mundo inteiro e um
//! pormenor). SCENE (por omissão o autosave), OUT (prefixo dos PNG),
//! CX/CY/SPAN (pormenor: centro em frações do mundo e largura em células).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{Scene, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let (extra, notes) = w.load_scene(&gpu, &scene).unwrap();
    println!("cena {path}: epoch {}", w.params.epoch);
    for n in notes {
        println!("  nota: {n}");
    }
    println!("mundo: fluido {}, contacto {}", w.settings.fluid_enabled, w.settings.contact_enabled);
    let changed: Vec<String> = w.params.changed_from_default().iter().filter(|c| c.0 != "epoch").map(|(n, v, _)| format!("{n}={v}")).collect();
    println!("parametros: {}", changed.join(" "));
    if let Some(b) = scene.extra_block("estatisticas") {
        let h = ribossome::stats::History::from_saved(&extra["estatisticas"], b);
        let rows = h.last_rows(30);
        if !rows.is_empty() {
            println!("estatisticas: média das últimas {} amostras (de {})", rows.len(), h.len());
            for (i, name) in h.names.iter().enumerate() {
                let m = rows.iter().map(|(_, v)| v[i] as f64).sum::<f64>() / rows.len() as f64;
                println!("  {name}\t{m:.3}");
            }
        }
    }
    if let Ok(out) = std::env::var("OUT") {
        let size = 1024;
        let cap = Capture::new(&gpu, &w, size);
        let s = cfg.sim_size();
        let rgba = cap.render(&gpu, &w, &Camera { center: [0.5 * s, 0.5 * s], zoom: size as f32 / s }, 0, 0.5);
        cap.save_png(&rgba, std::path::Path::new(&format!("{out}_mundo.png"))).unwrap();
        let span = envf("SPAN", 120.0) * cfg.world_units_per_cell as f32;
        let cam = Camera { center: [envf("CX", 0.5) * s, envf("CY", 0.5) * s], zoom: size as f32 / span };
        let rgba = cap.render(&gpu, &w, &cam, 0, 0.5);
        cap.save_png(&rgba, std::path::Path::new(&format!("{out}_pormenor.png"))).unwrap();
    }
}
