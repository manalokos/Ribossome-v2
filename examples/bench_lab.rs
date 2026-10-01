//! Benchmark do modo laboratório com população: ms por passo à medida que
//! a população cresce, e com sistemas desligados. SEEDS, CAP.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn steps(gpu: &Gpu, world: &mut World, n: u32) -> f64 {
    let t = std::time::Instant::now();
    let mut done = 0;
    while done < n {
        let k = MAX_STEPS_PER_FRAME.min(n - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        done += k;
    }
    gpu.wait_idle();
    t.elapsed().as_secs_f64() * 1000.0 / n as f64
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { grid_size: 1024, fluid_size: 512, max_agents: env("CAP", 100_000), ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    world.configure_lab();
    world.seed_lab(&gpu, 1, 1.5);
    let mut rng = ribossome::life::SplitMix(3);
    world.request_seeds(&ribossome::life::seed_requests(env("SEEDS", 20_000), [30, 200], true, cfg.sim_size(), &mut rng));
    for round in 0..env("ROUNDS", 4) {
        let ms = steps(&gpu, &mut world, 256);
        let alive = world.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).count();
        println!("ronda {round}: {ms:7.3} ms/passo   ({alive} agentes)");
    }
    let rft = world.params.rft_enabled;
    world.params.rft_enabled = 0;
    println!("sem RFT:        {:7.3} ms/passo", steps(&gpu, &mut world, 128));
    world.params.rft_enabled = rft;
    println!("com RFT outra vez: {:7.3} ms/passo", steps(&gpu, &mut world, 128));
}
