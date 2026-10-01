//! Teto de agentes: para várias capacidades, enche o mundo (sementes em
//! lotes + nascimentos) e mede o custo por passo com a população cheia.
//! Uso: cargo run --release --example bench_cap  [CAPS=60000,200000,500000]
use ribossome::gpu::Gpu;
use ribossome::params::{Agent, WorldConfig};
use ribossome::world::{MAX_SPAWN_REQUESTS, MAX_STEPS_PER_FRAME, World};

fn steps(gpu: &Gpu, world: &mut World, n: u32) {
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_steps(&gpu.queue, &mut enc, n);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let caps: Vec<u32> = std::env::var("CAPS")
        .unwrap_or_else(|_| "60000,200000,500000".into())
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect();
    let per_agent = size_of::<Agent>() + 64 + 64 + 64 * 8 + 64 * 4 * 4 + 4 * 4 + 16;
    for cap in caps {
        let cfg = WorldConfig { max_agents: cap, ..WorldConfig::DEFAULT };
        let mut world = World::new(&gpu, cfg, 1);
        world.seed_matter(&gpu, 1);
        let mut rng = ribossome::life::SplitMix(9);
        let mut alive = 0usize;
        for round in 0..200 {
            let reqs =
                ribossome::life::seed_requests(MAX_SPAWN_REQUESTS as u32, [30, 120], false, cfg.sim_size(), &mut rng);
            world.request_seeds(&reqs);
            steps(&gpu, &mut world, MAX_STEPS_PER_FRAME);
            if round % 10 == 9 {
                alive = world.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).count();
                if alive as u32 >= cap * 95 / 100 {
                    break;
                }
            }
        }
        // Medição com a população atual (sem sementes).
        let t = std::time::Instant::now();
        for _ in 0..10 {
            steps(&gpu, &mut world, MAX_STEPS_PER_FRAME);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / (10 * MAX_STEPS_PER_FRAME) as f64;
        let alive_end = world.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).count();
        let l = world.ledger_blocking(&gpu);
        println!(
            "capacidade {cap:>7}: {alive:>7} -> {alive_end:>7} vivos   {ms:6.3} ms/passo   memória dos agentes {:.0} MB   matéria presa {:.1}% do total",
            (cap as usize * per_agent) as f64 / 1e6,
            l.held_total() as f64 / l.total() as f64 * 100.0
        );
    }
}
