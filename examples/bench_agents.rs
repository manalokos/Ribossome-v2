//! Custo por passo com uma população ESTÁVEL: N nadadores construídos (sem
//! morte, fome nem reprodução) no terreno do projeto. PUSH, N, FLUID.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { fluid_size: env("FLUIDRES", 1024u32), ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    world.load_terrain_png(std::path::Path::new("assets/terreno.png")).unwrap();
    world.seed_matter(&gpu, 1);
    world.settings.fluid_enabled = env("FLUID", 1) != 0;
    world.params.agent_fluid_push = env("PUSH", world.params.agent_fluid_push);
    world.params.death_probability = 0.0;
    world.params.maintenance_cost = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.spawn_energy = 1000.0;
    let text = format!("AUG UGU UCU {} UAA", "GGU GCU CUG ".repeat(15));
    let g: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    let s = cfg.sim_size();
    let mut rng = ribossome::life::SplitMix(5);
    let n: u32 = env("N", 20000);
    let reqs: Vec<SpawnRequest> =
        (0..n).map(|_| SpawnRequest::with_genome(s * (0.05 + 0.9 * rng.f32()), s * (0.3 + 0.65 * rng.f32()), &g)).collect();
    world.request_seeds(&reqs);
    let run = |w: &mut World, steps: u32| {
        let t = std::time::Instant::now();
        let mut done = 0;
        while done < steps {
            let k = MAX_STEPS_PER_FRAME.min(steps - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            done += k;
        }
        gpu.wait_idle();
        t.elapsed().as_secs_f64() * 1000.0 / steps as f64
    };
    run(&mut world, 256);
    let ms = run(&mut world, 1024);
    let alive = world.life_counters_blocking(&gpu).alive(cfg.max_agents);
    println!("{} agentes, empurrão {}, fluido {}: {ms:.3} ms/passo", alive, world.params.agent_fluid_push, world.settings.fluid_enabled as u32);
}
