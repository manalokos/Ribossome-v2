//! Velocidade de natação em função do período do relógio: o nadador
//! construído (relógio + 15 glicinas), com todas as variantes do relógio
//! forçadas ao mesmo período. Sem fluido, browniano, contacto nem nascimentos.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

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

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let text = format!("AUG CAU CUU {} UAA", "GGU ".repeat(15));
    let g: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    for period in [160.0f32, 80.0, 40.0, 20.0, 12.0, 8.0, 5.0] {
        let cfg = WorldConfig { grid_size: 1024, fluid_size: 512, max_agents: 20_000, ..WorldConfig::DEFAULT };
        let mut world = World::new(&gpu, cfg, 3);
        world.configure_lab();
        world.seed_lab(&gpu, 3, 6.0);
        let mut table = world.organ_table.clone();
        for row in table.iter_mut().filter(|r| r.tipo == 5) {
            for v in row.variantes.iter_mut() {
                v.insert("periodo".into(), period);
            }
        }
        world.set_organ_table(&gpu.queue, table);
        world.settings.fluid_enabled = false;
        world.settings.contact_enabled = false;
        world.params.brownian = 0.0;
        world.params.phoretic_gain = 0.0;
        world.params.pairing_rate = 0.0;
        world.params.death_probability = 0.0;
        world.params.maintenance_cost = 0.0;
        world.params.sedimentation = 0.0;
        world.params.spawn_energy = 1000.0;
        world.params.inertia = std::env::var("INERTIA").ok().and_then(|v| v.parse().ok()).unwrap_or(world.params.inertia);
        world.params.motion_cost = std::env::var("MC").ok().and_then(|v| v.parse().ok()).unwrap_or(world.params.motion_cost);
        let s = cfg.sim_size();
        let mut rng = ribossome::life::SplitMix(5);
        let reqs: Vec<SpawnRequest> = (0..200)
            .map(|_| SpawnRequest::with_genome(s * (0.1 + 0.8 * rng.f32()), s * (0.3 + 0.6 * rng.f32()), &g))
            .collect();
        world.request_seeds(&reqs);
        run(&gpu, &mut world, 150);
        let start = world.read_agents_blocking(&gpu);
        let p0: std::collections::HashMap<u32, (f32, f32)> =
            start.iter().filter(|a| a.alive != 0).map(|a| (a.id, (a.pos_x, a.pos_y))).collect();
        let mean_e = |v: &[ribossome::params::Agent]| {
            let a: Vec<f32> = v.iter().filter(|a| a.alive != 0).map(|a| a.energy).collect();
            a.iter().sum::<f32>() / a.len().max(1) as f32
        };
        let e0 = mean_e(&start);
        if std::env::var("DEBUG").is_ok() {
            let alive = start.iter().filter(|a| a.alive != 0).count();
            let c = world.life_counters_blocking(&gpu);
            eprintln!("vivos {alive}, {c:?}");
        }
        run(&gpu, &mut world, 600);
        let end = world.read_agents_blocking(&gpu);
        let e1 = mean_e(&end);
        let d: Vec<f32> = end
            .iter()
            .filter(|a| a.alive != 0)
            .filter_map(|a| p0.get(&a.id).map(|&(x, y)| ((a.pos_x - x).powi(2) + (a.pos_y - y).powi(2)).sqrt()))
            .collect();
        let m = d.iter().sum::<f32>() / d.len().max(1) as f32;
        println!(
            "período {period:5.0}: deslocamento em 600 passos {m:7.1}, energia gasta {:.4}/passo (manutenção de 17 resíduos: {:.4})",
            (e0 - e1) / 600.0,
            0.002 * 17.0
        );
    }
}
