//! CONDENSADOR e DIFERENCIADOR (modos 4 e 5 do relé). Três agentes parados:
//!   bias em α + condensador α -> γ: deve dar pulsos regulares em γ;
//!   bias em α + diferenciador α -> γ: γ ~ 0 (a entrada não muda);
//!   relógio em α + diferenciador α -> γ: γ oscila (segue as mudanças).
//! Mostra α e γ no resíduo do relé de 4 em 4 passos.
use ribossome::gpu::Gpu;
use ribossome::life::organs::translate_organs;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

const B: [&str; 4] = ["A", "U", "G", "C"];

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(480);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 3);
    let code = ribossome::life::table::code_to_gpu(&w.organ_code);
    // 3.º codão do relé com índice 24: entrada α (bits 0–1 = 0), saída γ
    // (bits 2–3 = 2), força ×1 (bits 4–5 = 1).
    let mut relay_codon = None;
    let mut unit = None;
    let mut seen: Vec<u8> = Vec::new();
    for i in 0..64usize {
        let cod = format!("{}{}{}", B[i / 16], B[(i / 4) % 4], B[i % 4]);
        let body = translate_organs(&bases(&format!("AUG CAU AAA {cod} GGU UAA")), true, &code);
        if body.len() == 3 {
            if let Some((6, _, g)) = body[1].organ {
                seen.push(g);
                // entrada α (bits 0–1 = 0), saída γ (bits 2–3 = 2), qualquer força.
                if g & 3 == 0 && (g >> 2) & 3 == 2 && relay_codon.is_none() {
                    relay_codon = Some((cod.clone(), g));
                }
                if (30..=34).contains(&g) && unit.is_none() {
                    unit = Some(cod.clone());
                }
            }
        }
    }
    seen.sort();
    seen.dedup();
    println!("índices de intensidade que existem: {seen:?}");
    let (rc, gi) = relay_codon.expect("sem codão com entrada α e saída γ");
    println!("relé: codão {rc}, índice {gi} (força ×{})", [0.5, 1.0, 2.0, 4.0][((gi >> 4) & 3) as usize]);
    let one = unit.expect("sem codão de intensidade ~1");
    // H + S = bias (variante 0), H + L = relógio (variante 0), H + K = relé
    // condensador (variante 4), H + Y = relé diferenciador (variante 5).
    let make = |src: &str, relay: &str| bases(&format!("AUG CAU {src} {one} GGU CAU {relay} {rc} GGU GGU GGU UAA"));
    let cases = [
        ("bias + condensador", make("UCU", "AAA")), ("bias + diferenciador", make("UCU", "UAU")), ("relógio + diferenciador", make("CUU", "UAU")),
        // H + G = relé flip-flop (variante 1): liga com α > limiar, desliga
        // com α < −limiar; entre os dois, γ fica como estava.
        ("relógio + flip-flop", make("CUU", "GGU")),
    ];
    for (name, g) in &cases {
        let body = translate_organs(g, true, &code);
        let desc: Vec<String> = body.iter().enumerate().filter_map(|(k, r)| r.organ.map(|(t, p, gi)| format!("{k}: {}", ribossome::life::organs::describe(t, p, gi, &w.organ_table)))).collect();
        println!("{name}: {}", desc.join(" | "));
    }
    w.custom_terrain = Some((vec![0; cfg.cells() as usize], vec![0.0; cfg.cells() as usize]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.brownian_rot = 0.0;
    w.params.uv_strength = 0.0;
    w.params.sedimentation = 0.0;
    w.params.signal_crosstalk = 0.0;
    w.params.spawn_energy = 8.0;
    let reqs: Vec<SpawnRequest> = cases.iter().enumerate().map(|(i, (_, g))| SpawnRequest::with_genome(4000.0 + 3000.0 * i as f32, 6000.0, g)).collect();
    w.request_seeds(&reqs);
    let mut rows: Vec<Vec<(f32, f32)>> = vec![Vec::new(); cases.len()];
    let mut slots: Vec<Option<usize>> = vec![None; cases.len()];
    for _ in 0..steps / 4 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, 4);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        if slots[0].is_none() {
            let agents = w.read_agents_blocking(&gpu);
            for (i, s) in slots.iter_mut().enumerate() {
                let x = 4000.0 + 3000.0 * i as f32;
                *s = agents.iter().position(|a| a.alive != 0 && (a.pos_x - x).abs() < 400.0 && (a.pos_y - 6000.0).abs() < 400.0);
            }
        }
        let sig: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
        for (i, s) in slots.iter().enumerate() {
            if let Some(slot) = s {
                // O relé é o resíduo 3 (M, fonte, G, relé).
                let v = sig[slot * 64 + 3];
                rows[i].push((v[0], v[2]));
            }
        }
    }
    let bar = |v: f32| match v {
        v if v > 0.6 => '#',
        v if v > 0.25 => '+',
        v if v > 0.05 => '.',
        v if v < -0.6 => '=',
        v if v < -0.25 => '-',
        v if v < -0.05 => ',',
        _ => ' ',
    };
    for (i, (name, _)) in cases.iter().enumerate() {
        let a: String = rows[i].iter().map(|r| bar(r.0)).collect();
        let g: String = rows[i].iter().map(|r| bar(r.1)).collect();
        let (gmin, gmax) = rows[i].iter().fold((f32::MAX, f32::MIN), |m, r| (m.0.min(r.1), m.1.max(r.1)));
        let amean = rows[i].iter().map(|r| r.0).sum::<f32>() / rows[i].len().max(1) as f32;
        println!("\n{name}: α médio {amean:+.2}; γ entre {gmin:+.2} e {gmax:+.2}\n  α |{a}|\n  γ |{g}|");
    }
    println!("\n(cada carácter = 4 passos; # > 0,6  + > 0,25  . > 0,05; = - , os mesmos em negativo)");
}
