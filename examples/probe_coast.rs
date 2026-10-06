//! Um agente que pára de bater continua a deslizar? Nadadores DESENHADOS
//! (relógio + 15 glicinas) nadam 1500 passos; depois os relógios são
//! silenciados (clock_mute = 1) e mede-se a velocidade média (células por
//! passo) passo a passo. Sem inércia física, a translação devia acabar
//! assim que o corpo deixa de mudar de forma. GAIN = ganho da natação,
//! INERTIA = inércia dos pesados, LOAD = carga das juntas.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let code: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/codigo_orgaos.json").unwrap()).unwrap();
    // Relógio, variante 0 (promotor + modificador da tabela) + intensidade ×1.
    let mut clock = String::new();
    for (p, mods) in code.as_object().unwrap() {
        for (m, tv) in mods.as_object().unwrap() {
            if tv[0].as_u64() == Some(5) && tv[1].as_u64() == Some(0) {
                let cod = |l: &str| match l { "H" => "CAU", "A" => "GCU", "D" => "GAU", "L" => "CUU", "Q" => "CAA", "T" => "ACU", _ => panic!("codão {l}") };
                clock = format!("{}{}GAA", cod(p), cod(m));
            }
        }
    }
    let genome = bases(&format!("AUG{clock}{}UAA", "GGU".repeat(15)));
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let mut w = World::new(&gpu, cfg, 3);
    let n = cfg.cells() as usize;
    w.custom_terrain = Some((vec![0; n], vec![0.0; n]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.settings.terrain_enabled = false;
    w.settings.contact_enabled = false;
    w.params.uv_strength = 0.0;
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.motion_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.spawn_energy = 60.0;
    w.params.signal_mode = envf("MODE", 2.0);
    w.params.swim_gain = envf("GAIN", 10.0);
    w.params.sedimentation = 0.0;
    w.params.swim_grip = envf("GRIP", 2.0);
    w.params.inertia = envf("INERTIA", 0.0);
    w.params.joint_load = envf("LOAD", 1.0);
    let mut rng = ribossome::life::SplitMix(5);
    let s = cfg.sim_size();
    let reqs: Vec<SpawnRequest> = (0..200).map(|_| SpawnRequest::with_genome(s * (0.1 + 0.8 * rng.f32()), s * (0.1 + 0.8 * rng.f32()), &genome)).collect();
    w.request_seeds(&reqs);
    let step = |w: &mut World, k: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    for _ in 0..150 {
        step(&mut w, 10);
    }
    let wpc = cfg.world_units_per_cell as f32;
    let pos = |w: &World| -> std::collections::HashMap<u32, (f32, f32)> {
        w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).map(|a| (a.id, (a.pos_x, a.pos_y))).collect()
    };
    // Velocidade média por passo, medida passo a passo.
    let mut speed = |w: &mut World, steps: u32| -> Vec<f32> {
        let mut out = Vec::new();
        let mut p0 = pos(w);
        for _ in 0..steps {
            step(w, 1);
            let p1 = pos(w);
            let v: f32 = p1.iter().filter_map(|(id, b)| p0.get(id).map(|a| ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt())).sum::<f32>() / p1.len().max(1) as f32;
            out.push(v / wpc);
            p0 = p1;
        }
        out
    };
    // Avanço líquido (do princípio ao fim) em 600 passos.
    let a0 = pos(&w);
    for _ in 0..60 {
        step(&mut w, 10);
    }
    let a1 = pos(&w);
    let net: f32 = a1.iter().filter_map(|(id, b)| a0.get(id).map(|a| ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt())).sum::<f32>() / a1.len().max(1) as f32 / wpc;
    println!("aderência {}, ganho {}: avanço líquido {:.2} células em 600 passos", w.params.swim_grip, w.params.swim_gain, net);
    let before = speed(&mut w, 60);
    let v0 = before.iter().sum::<f32>() / before.len() as f32;
    w.params.clock_mute = 1.0;
    let after = speed(&mut w, 240);
    let at = |i: usize| after[i.min(after.len() - 1)];
    println!(
        "ganho {}, inércia {}, carga {}: a nadar {:.4} células/passo; depois de parar: passo 1 {:.4}, 5 {:.4}, 10 {:.4}, 20 {:.4}, 40 {:.4}, 80 {:.4}, 160 {:.4}, 240 {:.4}",
        w.params.swim_gain, w.params.inertia, w.params.joint_load, v0, at(0), at(4), at(9), at(19), at(39), at(79), at(159), at(239)
    );
    let half = after.iter().position(|&v| v < 0.5 * v0).map_or("nunca".to_string(), |i| (i + 1).to_string());
    let tenth = after.iter().position(|&v| v < 0.1 * v0).map_or("nunca".to_string(), |i| (i + 1).to_string());
    println!("   cai para metade ao passo {half}, para um décimo ao passo {tenth}");
}
