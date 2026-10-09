//! TERRENO POR RUÍDO: gera-o com várias sementes e opções e grava o mundo
//! inteiro (com as fumarolas marcadas), para ver as formas que dá.
//! OUT (prefixo dos PNG), SIZE (lado da imagem), GRID (lado do mundo).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::World;
use ribossome::world::terrain::NoiseTerrain;

fn main() {
    let out = std::env::var("OUT").unwrap_or_else(|_| "ruido".into());
    let size: u32 = std::env::var("SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let grid: u32 = std::env::var("GRID").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { grid_size: grid, fluid_size: grid / 2, max_agents: 4096, ..WorldConfig::DEFAULT };
    let mut w = World::new(&gpu, cfg, 1);
    let cap = Capture::new(&gpu, &w, size);
    cap.view.show_vents.set(true);
    let d = NoiseTerrain::default();
    let cases = [
        ("omissao", d),
        ("semente2", NoiseTerrain { seed: 2, ..d }),
        ("manchas_pequenas", NoiseTerrain { scale: 18.0, ..d }),
        ("mais_rocha", NoiseTerrain { rock: 0.48, rubble: 0.1, ..d }),
        ("sem_fundo", NoiseTerrain { depth: 0.0, vents: 12, ..d }),
    ];
    let side = cfg.sim_size();
    for (name, o) in cases {
        w.use_noise_terrain(&o);
        w.seed_matter(&gpu, 1);
        // Uns passos para as fontes das fumarolas chegarem à placa.
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, 2);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        let rgba = cap.render(&gpu, &w, &Camera { center: [side / 2.0, side / 2.0], zoom: size as f32 / side }, 0, 0.0);
        cap.save_png(&rgba, std::path::Path::new(&format!("{out}_{name}.png"))).unwrap();
        let g = w.read_gamma_blocking(&gpu);
        let rock = g.iter().filter(|&&v| v >= 3).count() as f32 / g.len() as f32 * 100.0;
        let rubble = g.iter().filter(|&&v| v > 0 && v < 3).count() as f32 / g.len() as f32 * 100.0;
        println!("{name}: rocha {rock:.1}%, entulho {rubble:.1}%");
    }
}
