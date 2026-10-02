//! Bioturbação: piscina do laboratório (sem física do terreno, por isso só
//! os agentes mexem os grãos) com uma faixa de ENTULHO a meio. Conta as
//! células de terreno que mudaram, confirma que os grãos e a matéria se
//! conservam. BIO = probabilidade (0 = desligada).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { grid_size: 1024, fluid_size: 512, max_agents: 100_000, ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    world.configure_lab();
    let n = cfg.grid_size as usize;
    // Faixa de entulho (1 grão) a meio, 200 células de altura.
    let mut g = vec![0u32; n * n];
    for y in n / 2 - 100..n / 2 + 100 {
        for x in 0..n {
            g[y * n + x] = 1;
        }
    }
    world.custom_terrain = Some((g.clone(), Vec::new()));
    let base = world.seed_lab(&gpu, 1, 1.5);
    world.params.bioturbation = env("BIO", 0.05);
    let mut rng = ribossome::life::SplitMix(3);
    world.request_seeds(&ribossome::life::seed_requests(20_000, [30, 200], true, cfg.sim_size(), &mut rng));
    let steps: u32 = env("STEPS", 3000);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        done += k;
    }
    gpu.wait_idle();
    let g2 = world.read_gamma_blocking(&gpu);
    let changed = g.iter().zip(&g2).filter(|(a, b)| a != b).count();
    let sum0: u64 = g.iter().map(|&v| v as u64).sum();
    let sum1: u64 = g2.iter().map(|&v| v as u64).sum();
    let max1 = g2.iter().max().unwrap();
    let led = world.ledger_blocking(&gpu);
    let alive = world.life_counters_blocking(&gpu).alive(cfg.max_agents);
    println!("bioturbação {}: {changed} células de terreno mudaram; grãos {sum0} -> {sum1}; máx/célula {max1}", world.params.bioturbation);
    println!("matéria {} -> {} ({} agentes)", base.total(), led.total(), alive);
}
