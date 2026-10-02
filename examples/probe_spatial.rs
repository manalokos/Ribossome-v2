//! Onde está a matéria ao longo do tempo? Densidade média (e % ativada) por
//! faixa de altura, na água e no entulho. TERRAIN (por omissão
//! assets/terreno.png), STEPS, REACT (reativação uniforme), SEEDS.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    let t: String = env("TERRAIN", String::from("assets/terreno.png"));
    if t != "-" {
        world.load_terrain_png(std::path::Path::new(&t)).unwrap();
    }
    world.seed_matter(&gpu, 1);
    world.params.reactivation_rate = env("REACT", 2e-5f32);
    let seeds: u32 = env("SEEDS", 0);
    if seeds > 0 {
        let mut rng = ribossome::life::SplitMix(3);
        world.request_seeds(&ribossome::life::seed_requests(seeds, [30, 200], true, cfg.sim_size(), &mut rng));
    }
    let total: u32 = env("STEPS", 100_000);
    let n = cfg.grid_size as usize;
    let bands = 8;
    let mut done = 0;
    let report = |world: &World, done: u32| {
        let cells = world.read_cells_blocking(&gpu);
        let g = world.read_gamma_blocking(&gpu);
        let mut line = format!("passo {done:>7}:");
        for b in (0..bands).rev() {
            let (mut wt, mut wa, mut wn, mut rt, mut rn) = (0u64, 0u64, 0u64, 0u64, 0u64);
            for y in b * n / bands..(b + 1) * n / bands {
                for x in 0..n {
                    let i = y * n + x;
                    let (mut a, mut s) = (0u64, 0u64);
                    for c in 0..4 {
                        a += (cells[i * 4 + c] & 0xFFFF) as u64;
                        s += (cells[i * 4 + c] >> 16) as u64;
                    }
                    if g[i] == 0 { wt += a + s; wa += a; wn += 1; } else if g[i] < 3 { rt += a + s; rn += 1; }
                }
            }
            line += &format!(" | {:.1}/{:.0}%/e{:.1}", wt as f64 / wn.max(1) as f64, wa as f64 / wt.max(1) as f64 * 100.0, rt as f64 / rn.max(1) as f64);
        }
        println!("{line}");
    };
    println!("(faixas de cima para baixo: água média/% ativada/entulho médio)");
    report(&world, 0);
    while done < total {
        let k = MAX_STEPS_PER_FRAME.min(total - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        done += k;
        if done % 20_000 == 0 {
            gpu.wait_idle();
            report(&world, done);
        }
    }
}
