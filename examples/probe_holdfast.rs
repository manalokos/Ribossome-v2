//! A ventosa prende? Nadadores DESENHADOS (relógio + 14 valinas), com e sem
//! ventosa na ponta, a cair por gravidade em dois mundos: com riscas verticais
//! de entulho (uma célula em cada 8) e só água. Mede quanto caem em STEPS
//! passos. Espera-se: com riscas, o que tem ventosa agarra-se e fica; o outro
//! continua a cair. Na água a ventosa não faz nada (caem o mesmo).
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn codon(l: &str) -> &'static str {
    match l {
        "A" => "GCU", "C" => "UGU", "D" => "GAU", "E" => "GAA", "F" => "UUU", "G" => "GGU", "H" => "CAU", "I" => "AUU", "K" => "AAA",
        "L" => "CUU", "N" => "AAU", "P" => "CCU", "Q" => "CAA", "R" => "CGU", "S" => "UCU", "T" => "ACU", "V" => "GUU", "W" => "UGG", "Y" => "UAU",
        _ => panic!("aminoácido {l}"),
    }
}

fn organ(code: &serde_json::Value, t: u64, v: u64) -> String {
    for (p, mods) in code.as_object().unwrap() {
        for (m, tv) in mods.as_object().unwrap() {
            if tv[0].as_u64() == Some(t) && tv[1].as_u64() == Some(v) {
                return format!("{}{}GAA", codon(p), codon(m));
            }
        }
    }
    panic!("órgão {t}/{v} não está na tabela");
}

fn bases(s: &str) -> Vec<u8> {
    s.chars().map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let code: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/codigo_orgaos.json").unwrap()).unwrap();
    // Valina: guarda mais energia do que a glicina (a capacidade vem do volume).
    let body = "GUU".repeat(14);
    let clock = organ(&code, 5, 0);
    let designs = [
        ("sem ventosa", format!("AUG{clock}{body}UAA")),
        ("com ventosa (força 1500)", format!("AUG{clock}{body}{}UAA", organ(&code, 18, 2))),
    ];
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let steps = envf("STEPS", 1500.0) as u32;
    // Mundos: riscas verticais de entulho (1 célula em cada 8), com
    // gravidade nos agentes; e só água, com a mesma gravidade.
    for (world_name, stripes) in [("riscas de entulho, com gravidade", true), ("só água, com gravidade", false)] {
        let mut w = World::new(&gpu, cfg, 3);
        let n = cfg.cells() as usize;
        let gs = cfg.grid_size as usize;
        let terrain: Vec<u32> = (0..n).map(|i| (stripes && (i % gs) % 8 == 0) as u32).collect();
        w.custom_terrain = Some((terrain, vec![0.0; n]));
        w.fumaroles.clear();
        w.seed_matter(&gpu, 3);
        w.settings.fluid_enabled = false;
        w.settings.contact_enabled = false;
        w.params.uv_strength = 0.0;
        w.params.death_probability = 0.0;
        w.params.maintenance_cost = 0.0;
        w.params.motion_cost = 0.0;
        w.params.pairing_rate = 0.0;
        w.params.uptake_rate = 0.0;
        w.params.bioturbation = 0.0;
        w.params.sedimentation = envf("GRAVITY", 0.3);
        w.params.spawn_energy = 60.0;
        w.params.signal_mode = 2.0;
        let mut rng = ribossome::life::SplitMix(5);
        let s = cfg.sim_size();
        let mut reqs = Vec::new();
        let mut by_genome: HashMap<Vec<u8>, usize> = HashMap::new();
        for (i, (_, g)) in designs.iter().enumerate() {
            let b = bases(g);
            by_genome.insert(b.clone(), i);
            for _ in 0..150 {
                reqs.push(SpawnRequest::with_genome(s * (0.1 + 0.8 * rng.f32()), s * (0.75 + 0.2 * rng.f32()), &b));
            }
        }
        w.request_seeds(&reqs);
        let run = |w: &mut World, k: u32| {
            let mut done = 0;
            while done < k {
                let c = MAX_STEPS_PER_FRAME.min(k - done);
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                w.encode_steps(&gpu.queue, &mut enc, c);
                gpu.queue.submit([enc.finish()]);
                gpu.wait_idle();
                done += c;
            }
        };
        let snap = |w: &World| -> HashMap<u32, (usize, f32, f32)> {
            let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
            let mut out = HashMap::new();
            for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
                let g: Vec<u8> = (0..a.gene_len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
                if let Some(&i) = by_genome.get(&g) {
                    out.insert(a.id, (i, a.pos_x, a.pos_y));
                }
            }
            out
        };
        run(&mut w, 100);
        let a0 = snap(&w);
        run(&mut w, steps);
        let a1 = snap(&w);
        let wpc = cfg.world_units_per_cell as f32;
        print!("{world_name}:");
        for (i, (name, _)) in designs.iter().enumerate() {
            // Queda (células para baixo) desde o passo 100.
            let d: Vec<f32> = a1.iter().filter(|(_, v)| v.0 == i).filter_map(|(id, b)| a0.get(id).map(|a| (a.2 - b.2) / wpc)).collect();
            print!("  {name}: {} agentes, caíram {:.1} células em {steps} passos;", d.len(), d.iter().sum::<f32>() / d.len().max(1) as f32);
        }
        println!();
    }
}
