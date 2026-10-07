//! Tempo de GPU de cada kernel, por passo, com o terreno do projeto e N
//! nadadores construídos estáveis (sem morte nem fome). N, PUSH, FLUIDRES.
//! Ou SCENE=cena.ribo para medir uma cena gravada.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { fluid_size: env("FLUIDRES", 1024u32), ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    world.load_terrain_png(std::path::Path::new("assets/terreno.png")).unwrap();
    world.seed_matter(&gpu, 1);
    world.params.agent_fluid_push = env("PUSH", world.params.agent_fluid_push);
    world.params.aggregation = env("AGG", world.params.aggregation);
    world.params.death_probability = 0.0;
    world.params.maintenance_cost = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.spawn_energy = 1000.0;
    world.params.rft_enabled = env("RFT", 1u32);
    world.params.monomer_pressure = env("PRESS", world.params.monomer_pressure);
    world.params.sedimentation = env("SED", world.params.sedimentation);
    world.params.uptake_rate = env("UPTAKE", world.params.uptake_rate);
    world.settings.fluid_enabled = env("FLUID", 1u32) != 0;
    let text = format!("AUG CAU CUU {} UAA", "GGU GCU CUG ".repeat(15));
    let g: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    let s = cfg.sim_size();
    let mut rng = ribossome::life::SplitMix(5);
    let n: u32 = env("N", 20000);
    // SCENE=cena.ribo: mede uma cena gravada tal como está (parâmetros,
    // terreno e agentes dela) em vez dos nadadores construídos.
    if let Ok(path) = std::env::var("SCENE") {
        let scene = ribossome::world::Scene::read(std::path::Path::new(&path)).unwrap();
        world.load_scene(&gpu, &scene).unwrap();
    } else {
        let reqs: Vec<SpawnRequest> =
            (0..n).map(|_| SpawnRequest::with_genome(s * (0.05 + 0.9 * rng.f32()), s * (0.3 + 0.65 * rng.f32()), &g)).collect();
        world.request_seeds(&reqs);
    }
    // PARAMS=nome=valor,... por cima de tudo (também da cena).
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(world.params.set_named(k.trim(), v.trim().parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
    }
    let step = |w: &mut World, k: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    for _ in 0..8 {
        step(&mut world, 32);
    }
    if !world.enable_kernel_timing(&gpu) {
        println!("esta GPU não tem timestamps dentro dos passes");
        return;
    }
    // 16 lotes de 8 passos (8 passos cobrem os passos com e sem fluido).
    let mut total: Vec<(&'static str, f64)> = Vec::new();
    let batches = 16;
    for _ in 0..batches {
        step(&mut world, 8);
        for (l, ms) in world.read_kernel_timing(&gpu) {
            match total.iter_mut().find(|(n, _)| *n == l) {
                Some(e) => e.1 += ms,
                None => total.push((l, ms)),
            }
        }
    }
    let steps = (batches * 8) as f64;
    total.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let sum: f64 = total.iter().map(|(_, m)| m).sum::<f64>() / steps;
    let alive = world.life_counters_blocking(&gpu).alive(cfg.max_agents);
    println!("{alive} agentes; soma dos kernels {sum:.3} ms/passo");
    for (l, ms) in total.iter().take(25) {
        println!("  {:<24} {:7.3} ms/passo  ({:4.1}%)", l, ms / steps, ms / steps / sum * 100.0);
    }
}
