//! Nadador construído à mão: relógio + músculos espaçados (uma onda pela
//! cadeia), comparado com o mesmo corpo sem relógio. Mede o deslocamento
//! próprio (sem fluido, sem browniano, sem contacto, sem nascimentos).
use ribossome::gpu::Gpu;
use ribossome::life::organs::translate_organs;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn run(gpu: &Gpu, world: &mut World, steps: u32) {
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
}

fn trial(gpu: &Gpu, name: &str, genome: &str) {
    let g = bases(genome);
    let body = translate_organs(&g, true, &ribossome::life::table::code_to_gpu(&ribossome::life::table::load_code().0));
    let organs: Vec<String> = body
        .iter()
        .filter_map(|r| r.organ.map(|(t, p, _)| format!("{}({p})", ribossome::life::organs::ORGAN_NAMES[t as usize])))
        .collect();
    // Laboratório: piscina 1024² sem fluido nem terreno.
    let cfg = WorldConfig { grid_size: 1024, fluid_size: 512, max_agents: 20_000, ..WorldConfig::DEFAULT };
    let mut world = World::new(gpu, cfg, 3);
    world.configure_lab();
    world.seed_lab(gpu, 3, 6.0);
    world.params.swim_gain = std::env::var("SWIM").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    world.params.thermal_kt = std::env::var("KT").ok().and_then(|v| v.parse().ok()).unwrap_or(world.params.thermal_kt);
    world.settings.fluid_enabled = false;
    world.settings.contact_enabled = false;
    world.params.brownian = 0.0;
    world.params.phoretic_gain = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.death_probability = 0.0;
    // Sem boca não comem: sem manutenção, para medir só a natação.
    world.params.maintenance_cost = 0.0;
    world.params.sedimentation = 0.0;
    let s = cfg.sim_size();
    let mut rng = ribossome::life::SplitMix(5);
    let reqs: Vec<SpawnRequest> = (0..200)
        .map(|_| SpawnRequest::with_genome(s * (0.1 + 0.8 * rng.f32()), s * (0.3 + 0.6 * rng.f32()), &g))
        .collect();
    world.request_seeds(&reqs);
    run(gpu, &mut world, 150);
    let p0: std::collections::HashMap<u32, (f32, f32)> =
        world.read_agents_blocking(gpu).iter().filter(|a| a.alive != 0).map(|a| (a.id, (a.pos_x, a.pos_y))).collect();
    let disp = |w: &World| -> f32 {
        let d: Vec<f32> = w
            .read_agents_blocking(gpu)
            .iter()
            .filter(|a| a.alive != 0)
            .filter_map(|a| p0.get(&a.id).map(|&(x, y)| ((a.pos_x - x).powi(2) + (a.pos_y - y).powi(2)).sqrt()))
            .collect();
        d.iter().sum::<f32>() / d.len().max(1) as f32
    };
    run(gpu, &mut world, 300);
    let m300 = disp(&world);
    run(gpu, &mut world, 300);
    let m600 = disp(&world);
    println!("{name:<28} {} resíduos, órgãos [{}]", body.len(), organs.join(", "));
    println!(
        "    deslocamento médio {m300:6.1} -> {m600:6.1} (x{:.2}; balístico x2)  [{} agentes]",
        m600 / m300.max(1e-6),
        p0.len()
    );
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let clock = "UGU UCU";
    let gly = format!("AUG {clock} {} UAA", "GGU ".repeat(15));
    trial(&gpu, "relógio + 15 glicinas", &gly);
    let gly_noclock = format!("AUG GCU GCU {} UAA", "GGU ".repeat(15));
    trial(&gpu, "15 glicinas, sem relógio", &gly_noclock);
}
