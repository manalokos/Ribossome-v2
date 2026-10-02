//! Os agentes empurram a água: velocidade média do fluido (água livre) com
//! PUSH = agent_fluid_push, sementes SEEDS, STEPS passos.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { max_agents: 200_000, ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    world.load_terrain_png(std::path::Path::new("assets/terreno.png")).unwrap();
    world.seed_matter(&gpu, 1);
    world.params.agent_fluid_push = env("PUSH", 0.2f32);
    let mut rng = ribossome::life::SplitMix(7);
    world.request_seeds(&ribossome::life::seed_requests(env("SEEDS", 20000), [30, 200], true, cfg.sim_size(), &mut rng));
    let steps: u32 = env("STEPS", 4000);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        done += k;
    }
    gpu.wait_idle();
    let vel = world.read_f32_blocking(&gpu, &world.velocity_buf);
    let (mut s, mut mx, mut nan) = (0f64, 0f32, 0);
    for v in vel.chunks(2) {
        if !v[0].is_finite() || !v[1].is_finite() { nan += 1; continue; }
        let m = v[0].hypot(v[1]);
        s += m as f64;
        mx = mx.max(m);
    }
    let alive = world.life_counters_blocking(&gpu).alive(cfg.max_agents);
    println!("push {}: |v| médio {:.3}, máx {:.1}, NaN {nan}, {alive} agentes", world.params.agent_fluid_push, s / (vel.len() / 2) as f64, mx);
}
