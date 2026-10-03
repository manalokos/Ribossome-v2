//! Aguentam a noite? Fotossintéticos desenhados (fotossistema CS + armazenamento
//! WE de capacidade 32, com intensidades diferentes) perto da superfície de um
//! mundo sem terreno, com o ciclo dia/noite. Sem reprodução. Conta os vivos e a
//! energia média ao fim do dia e ao fim da noite. PERIOD (epochs).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let period: f32 = std::env::var("PERIOD").ok().and_then(|v| v.parse().ok()).unwrap_or(20000.0);
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 2);
    let n = cfg.cells() as usize;
    w.custom_terrain = Some((vec![0; n], vec![0.0; n]));
    w.seed_matter(&gpu, 2);
    w.params.pairing_rate = 0.0;
    w.params.sedimentation = 0.0;
    w.params.day_period = period;
    w.params.spawn_energy = 1000.0; // começam cheios (limitado à capacidade)
    // (nome, intensidade do armazenamento)
    let kinds = [("armazenamento ×1", ""), ("armazenamento ×2,4", "GGG"), ("armazenamento ×14,7", "CCC")];
    let s = cfg.sim_size();
    let mut reqs = Vec::new();
    let mut groups = Vec::new();
    for (gi, (_, gain)) in kinds.iter().enumerate() {
        let text = format!("AUG UGU UCU CCC UGG GAA {gain} GGU GGU GGU GGU UAA");
        let g: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
        for i in 0..300 {
            let x = s * (0.05 + 0.9 * (i as f32 + 0.5) / 300.0);
            reqs.push(SpawnRequest::with_genome(x, s * 0.97, &g));
            groups.push(gi);
        }
    }
    w.request_seeds(&reqs);
    let run = |w: &mut World, steps: u32| {
        let mut done = 0;
        while done < steps {
            let k = MAX_STEPS_PER_FRAME.min(steps - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += k;
        }
    };
    run(&mut w, 8);
    // Identifica os grupos pelo id (sementes por ordem).
    let a0 = w.read_agents_blocking(&gpu);
    let mut ids: Vec<(u32, u32)> = a0.iter().filter(|a| a.alive != 0).map(|a| (a.id, a.gene_len)).collect();
    ids.sort();
    let group_of = |id: u32| -> usize { (id as usize).min(groups.len() - 1).min(groups.len() - 1) / 300 };
    let report = |w: &World, label: &str| {
        let a = w.read_agents_blocking(&gpu);
        let c = w.life_counters_blocking(&gpu);
        let temp = w.read_f32_blocking(&gpu, &w.temp_buf);
        let f = cfg.fluid_size as usize;
        let row = (0.97 * f as f32) as usize;
        let t_top = temp[row * f..(row + 1) * f].iter().sum::<f32>() / f as f32;
        let t_max = temp.iter().cloned().fold(0.0f32, f32::max);
        println!(
            "{label} (epoch {}, sol {:.2}): mortes {} (fome {}), T à superfície {:.2}, T máx {:.2}",
            w.params.epoch, w.params.daylight(w.params.epoch), c.deaths, c.starved, t_top, t_max
        );
        for (gi, (name, _)) in kinds.iter().enumerate() {
            let v: Vec<f32> = a.iter().filter(|x| x.alive != 0 && group_of(x.id) == gi).map(|x| x.energy).collect();
            println!("  {name:<20} vivos {:3}/300  energia média {:6.1}", v.len(), v.iter().sum::<f32>() / v.len().max(1) as f32);
        }
    };
    let p = period as u32;
    run(&mut w, p / 2 - 8);
    report(&w, "fim do dia");
    run(&mut w, p / 2);
    report(&w, "fim da noite");
    let _ = ids;
}
