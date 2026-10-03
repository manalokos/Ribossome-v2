//! Balanço lateral com fluido: 300 nadadores construídos (relógio +
//! glicinas) no mundo completo sem terreno; caminho percorrido vs
//! deslocamento líquido em STEPS passos. PUSH = agent_fluid_push.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { max_agents: 4096, ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    world.custom_terrain = Some((vec![0; (cfg.grid_size * cfg.grid_size) as usize], vec![0.0; (cfg.grid_size * cfg.grid_size) as usize]));
    world.seed_matter(&gpu, 1);
    world.params.agent_fluid_push = env("PUSH", 0.5f32);
    world.params.death_probability = 0.0;
    world.params.maintenance_cost = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.sedimentation = 0.0;
    world.params.spawn_energy = env("ENERGY", 5.0f32);
    world.params.spawn_energy = env("ENERGY", 5.0f32);
    world.settings.fluid_enabled = env("FLUID", 1) != 0;
    world.params.rft_enabled = env("RFT", 1);
    world.params.swim_wobble = env("WOBBLE", 1.0f32);
    world.params.fluid_swim_only = env("FSO", 0u32);
    let control = env("CONTROL", 0u32) != 0;
    // Controlo: o mesmo corpo sem relógio (não bate, não deve nadar).
    let g = if control {
        bases(&format!("AUG {} UAA", "GGU ".repeat(16)))
    } else {
        bases(&format!("AUG CAU CUU {} UAA", "GGU ".repeat(15)))
    };
    let s = cfg.sim_size();
    let mut rng = ribossome::life::SplitMix(5);
    let reqs: Vec<SpawnRequest> = (0..300)
        .map(|_| SpawnRequest::with_genome(s * (0.2 + 0.6 * rng.f32()), s * (0.3 + 0.5 * rng.f32()), &g))
        .collect();
    world.request_seeds(&reqs);
    let run = |w: &mut World, n: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, n);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    run(&mut world, 200);
    let start = world.read_agents_blocking(&gpu);
    let mut prev = start.clone();
    // Persistência: cosseno entre deslocamentos de janelas seguidas de 100 passos.
    let mut win_start = start.clone();
    let mut last_disp: Vec<Option<(f64, f64)>> = vec![None; start.len()];
    let (mut cos_sum, mut cos_n) = (0f64, 0f64);
    let mut tick = 0u32;
    let mut path = vec![0f64; start.len()];
    for _ in 0..env("STEPS", 1000) / 10 {
        run(&mut world, 10);
        let now = world.read_agents_blocking(&gpu);
        for (i, (a, b)) in prev.iter().zip(&now).enumerate() {
            if a.alive != 0 && b.alive != 0 && a.id == b.id {
                path[i] += ((b.pos_x - a.pos_x) as f64).hypot((b.pos_y - a.pos_y) as f64);
            }
        }
        tick += 1;
        if tick.is_multiple_of(10) {
            for (i, (a, b)) in win_start.iter().zip(&now).enumerate() {
                if a.alive == 0 || b.alive == 0 || a.id != b.id { last_disp[i] = None; continue; }
                let d = ((b.pos_x - a.pos_x) as f64, (b.pos_y - a.pos_y) as f64);
                let l = d.0.hypot(d.1);
                if let Some(pd) = last_disp[i] {
                    let pl = pd.0.hypot(pd.1);
                    if l > 1e-3 && pl > 1e-3 { cos_sum += (d.0 * pd.0 + d.1 * pd.1) / (l * pl); cos_n += 1.0; }
                }
                last_disp[i] = Some(d);
            }
            win_start = now.clone();
        }
        prev = now;
    }
    let (mut p, mut net, mut n) = (0f64, 0f64, 0f64);
    for (i, (a, b)) in start.iter().zip(&prev).enumerate() {
        if a.alive != 0 && b.alive != 0 && a.id == b.id {
            p += path[i];
            net += ((b.pos_x - a.pos_x) as f64).hypot((b.pos_y - a.pos_y) as f64);
            n += 1.0;
        }
    }
    print!(
        "{} fluido {} natação {} só-fluido {} vaivém {} ",
        if control { "CONTROLO" } else { "nadador" },
        world.settings.fluid_enabled as u32,
        world.params.rft_enabled,
        world.params.fluid_swim_only,
        world.params.swim_wobble
    );
    println!(
        "push {}: caminho médio {:.0}, deslocamento líquido {:.0}, persistência da direção {:+.2} ({} agentes)",
        world.params.agent_fluid_push, p / n, net / n, cos_sum / cos_n.max(1.0), n
    );
}
