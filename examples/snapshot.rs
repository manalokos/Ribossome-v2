//! Corre o mundo sem janela e grava PNGs da vista (para depurar sem ecrã).
//! Variáveis: STEPS (lista separada por vírgulas), VIEWS (ex. "0,7,9"),
//! FLUID=0, FLAT=1, BIG=1, ZOOM (1 = mundo inteiro), CX/CY (centro, fração), OUT (pasta).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let mut cfg = if env("BIG", 1) == 1 { WorldConfig::DEFAULT } else { WorldConfig::TEST };
    cfg.fluid_size = env("FLUIDRES", cfg.fluid_size);
    let mut world = World::new(&gpu, cfg, 1);
    world.settings.fluid_enabled = env("FLUID", 1) == 1;
    world.params.cohesion = env("COH", world.params.cohesion);
    world.seed_matter(&gpu, 1);
    if env("FLAT", 0) == 1 {
        gpu.queue.write_buffer(&world.gamma_buf, 0, &vec![0u8; (cfg.cells() * 4) as usize]);
    }
    let seeds: u32 = env("SEEDS", 0);
    if seeds > 0 {
        let mut rng = ribossome::life::SplitMix(7);
        let reqs = ribossome::life::seed_requests(seeds, [12, 120], true, cfg.sim_size(), &mut rng);
        world.request_seeds(&reqs);
    }
    let out = std::path::PathBuf::from(env("OUT", String::from("target/snapshots")));
    std::fs::create_dir_all(&out).unwrap();
    let size = 1024;
    let cap = Capture::new(&gpu, &world, size);
    let s = cfg.sim_size();
    let zoom: f32 = env("ZOOM", 1.0);
    let cam = Camera { center: [env("CX", 0.5f32) * s, env("CY", 0.5f32) * s], zoom: size as f32 / s * zoom };
    let views: Vec<u32> = env("VIEWS", String::from("0")).split(',').filter_map(|v| v.parse().ok()).collect();
    let mut marks: Vec<u32> = env("STEPS", String::from("0,2000")).split(',').filter_map(|v| v.parse().ok()).collect();
    marks.sort();
    let mut done = 0u32;
    for m in marks {
        while done < m {
            let k = MAX_STEPS_PER_FRAME.min(m - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            world.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += k;
        }
        for &v in &views {
            let rgba = cap.render(&gpu, &world, &cam, v, env("BRIGHT", 0.5));
            let p = out.join(format!("passo{m:06}_vista{v}.png"));
            cap.save_png(&rgba, &p).unwrap();
            println!("{}", p.display());
        }
    }
}
