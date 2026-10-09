//! MAPA DE PARENTESCO: carrega uma cena, toma o genoma mais comum como
//! referência e grava a vista com a cor dos agentes = parentesco (o mapa de
//! Voronoi do fundo), de perto e de longe. SCENE, OUT (prefixo), SIZE.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::species::{cluster, living_genomes};
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let out = std::env::var("OUT").unwrap_or_else(|_| "parentesco".into());
    let size: u32 = std::env::var("SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let genomes = living_genomes(&gpu, &w);
    let species = cluster(&genomes, 0.15);
    w.set_kin_target(&gpu.queue, &species[0].leader);
    // Uns passos (a grelha de contacto é a da simulação) e o parentesco de todos.
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    w.encode_steps(&gpu.queue, &mut enc, 4);
    w.encode_kinship(&mut enc);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let agents = w.read_agents_blocking(&gpu);
    let alive: Vec<_> = agents.iter().filter(|a| a.alive != 0).collect();
    let centre = alive
        .iter()
        .step_by((alive.len() / 300).max(1))
        .max_by_key(|a| alive.iter().filter(|b| (a.pos_x - b.pos_x).abs() < 600.0 && (a.pos_y - b.pos_y).abs() < 600.0).count())
        .map(|a| [a.pos_x, a.pos_y])
        .unwrap();
    let cap = Capture::new(&gpu, &w, size);
    cap.signal_view.set(4);
    for (name, width) in [("perto", 1500.0f32), ("medio", 6000.0), ("longe", 24000.0)] {
        let rgba = cap.render(&gpu, &w, &Camera { center: centre, zoom: size as f32 / width }, 0, 0.0);
        cap.save_png(&rgba, std::path::Path::new(&format!("{out}_{name}.png"))).unwrap();
    }
    println!("referência: a espécie mais comum ({} de {} agentes)", species[0].count, genomes.len());
}
