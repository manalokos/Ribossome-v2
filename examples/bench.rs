//! Benchmark do passo do mundo: liga e desliga sistemas para ver onde vai o
//! tempo. Mede ms por passo (lotes de 64 passos, com espera no fim).
//! Uso: cargo run --release --example bench
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn measure(gpu: &Gpu, world: &mut World, steps: u32) -> f64 {
    // Aquecimento.
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let t = std::time::Instant::now();
    let mut done = 0;
    while done < steps {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
        gpu.queue.submit([enc.finish()]);
        done += MAX_STEPS_PER_FRAME;
    }
    gpu.wait_idle();
    t.elapsed().as_secs_f64() * 1000.0 / done as f64
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let steps = 640;
    let mut run = |name: &str, cfg: WorldConfig, f: &dyn Fn(&mut World), seeds: u32| {
        let mut world = World::new(&gpu, cfg, 1);
        world.seed_matter(&gpu, 1);
        f(&mut world);
        if seeds > 0 {
            let mut rng = ribossome::life::SplitMix(3);
            let reqs = ribossome::life::seed_requests(seeds, [12, 120], true, cfg.sim_size(), &mut rng);
            world.request_seeds(&reqs);
        }
        let ms = measure(&gpu, &mut world, steps);
        let alive = world.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).count();
        println!("{name:<40} {ms:6.3} ms/passo   ({alive} agentes)");
    };
    let big = WorldConfig::DEFAULT;
    run("completo", big, &|_| {}, 0);
    run("sem fluido", big, &|w| w.settings.fluid_enabled = false, 0);
    run("pressão por Jacobi 128 (v3)", big, &|w| w.settings.multigrid = false, 0);
    run("sem física do terreno", big, &|w| w.settings.terrain_enabled = false, 0);
    run(
        "sem fluido nem terreno (só química)",
        big,
        &|w| {
            w.settings.fluid_enabled = false;
            w.settings.terrain_enabled = false;
        },
        0,
    );
    run("fluido 512²", WorldConfig { fluid_size: 512, ..big }, &|_| {}, 0);
    run("completo + 4000 sementes", big, &|_| {}, 4000);
}
