//! O órgão de dormência baixa mesmo o metabolismo? Corpos DESENHADOS num
//! mundo sem comida, sem morte e sem custo de movimento: a única despesa é
//! a manutenção (× metabolismo). Mede a energia gasta por passo em cada
//! desenho:
//! - controlo: 12 glicinas;
//! - dormência fixa ×0,7 e ×0,4 (variantes 0 e 1);
//! - dormência por γ (variante 3, ×0,25) SEM sinal: não deve fazer nada;
//! - a mesma COM sinal: um bias em β e um relé cópia β→γ antes do órgão.
//! Modo de sinais 2 (o sinal anda do lado N para o C).
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn codon(l: char) -> &'static str {
    match l {
        'A' => "GCU", 'C' => "UGU", 'D' => "GAU", 'E' => "GAA", 'F' => "UUU", 'G' => "GGU", 'H' => "CAU",
        'I' => "AUU", 'K' => "AAA", 'L' => "CUU", 'M' => "AUG", 'N' => "AAU", 'P' => "CCU", 'Q' => "CAA",
        'R' => "CGU", 'S' => "UCU", 'T' => "ACU", 'V' => "GUU", 'W' => "UGG", 'Y' => "UAU",
        _ => panic!("aminoácido {l}"),
    }
}

/// Promotor + modificador (da tabela) + 3.º codão (GAA = intensidade ×1).
fn organ(code: &serde_json::Value, t: u64, v: u64, third: &str) -> String {
    for (p, mods) in code.as_object().unwrap() {
        for (m, tv) in mods.as_object().unwrap() {
            if tv[0].as_u64() == Some(t) && tv[1].as_u64() == Some(v) {
                return format!("{}{}{third}", codon(p.chars().next().unwrap()), codon(m.chars().next().unwrap()));
            }
        }
    }
    panic!("órgão {t}/{v} não está na tabela");
}

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let code: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/codigo_orgaos.json").unwrap()).unwrap();
    let organs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/orgaos.json").unwrap()).unwrap();
    // Bias em β com valor positivo.
    let bias_v = organs[13]["variantes"]
        .as_array()
        .unwrap()
        .iter()
        .position(|v| v["canal"].as_f64() == Some(1.0) && v["valor"].as_f64().unwrap_or(0.0) > 0.0)
        .expect("bias β positivo") as u64;
    let body = codon('G').repeat(12);
    let bias = organ(&code, 13, bias_v, "GAA");
    // Relé cópia (variante 1), 3.º codão UGU: entrada β, saída γ, força ×1.
    let relay = organ(&code, 6, 1, "UGU");
    let designs = [
        ("controlo (12 glicinas)", format!("AUG{body}UAA")),
        ("dormência fixa ×0,7", format!("AUG{}{body}UAA", organ(&code, 16, 0, "GAA"))),
        ("dormência fixa ×0,4", format!("AUG{}{body}UAA", organ(&code, 16, 1, "GAA"))),
        ("dormência por γ ×0,25, sem sinal", format!("AUG{}{body}UAA", organ(&code, 16, 3, "GAA"))),
        ("bias β + relé β→γ, sem dormência", format!("AUG{bias}{relay}{body}UAA")),
        ("bias β + relé β→γ + dormência por γ ×0,25", format!("AUG{bias}{relay}{}{body}UAA", organ(&code, 16, 3, "GAA"))),
    ];
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let mut w = World::new(&gpu, cfg, 3);
    let n = cfg.cells() as usize;
    w.custom_terrain = Some((vec![0; n], vec![0.0; n]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.settings.terrain_enabled = false;
    w.params.uv_strength = 0.0;
    w.params.death_probability = 0.0;
    w.params.motion_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.spawn_energy = 60.0;
    w.params.signal_mode = 2.0;
    let mut rng = ribossome::life::SplitMix(5);
    let s = cfg.sim_size();
    let mut reqs = Vec::new();
    let mut by_genome: HashMap<Vec<u8>, usize> = HashMap::new();
    for (i, (_, g)) in designs.iter().enumerate() {
        let b = bases(g);
        by_genome.insert(b.clone(), i);
        for _ in 0..100 {
            reqs.push(SpawnRequest::with_genome(s * (0.05 + 0.9 * rng.f32()), s * (0.05 + 0.9 * rng.f32()), &b));
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
    let energies = |w: &World| -> Vec<(f32, u32, u32)> {
        let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
        let mut out = vec![(0f32, 0u32, 0u32); designs.len()];
        for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
            let g: Vec<u8> = (0..a.gene_len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
            if let Some(&i) = by_genome.get(&g) {
                out[i].0 += a.energy;
                out[i].1 += 1;
                out[i].2 = a.body_len;
            }
        }
        out
    };
    // Os primeiros passos são o nascimento e a dobragem: mede-se depois.
    run(&mut w, 20);
    let e0 = energies(&w);
    let steps = 100;
    run(&mut w, steps);
    let e1 = energies(&w);
    let base = (e0[0].0 / e0[0].1.max(1) as f32 - e1[0].0 / e1[0].1.max(1) as f32) / steps as f32;
    println!("energia gasta por passo (mundo sem comida; {steps} passos):");
    for (i, (name, _)) in designs.iter().enumerate() {
        let (a, b) = (e0[i].0 / e0[i].1.max(1) as f32, e1[i].0 / e1[i].1.max(1) as f32);
        let loss = (a - b) / steps as f32;
        println!("  {name:44} {:3} agentes, {:2} resíduos: {:.5} por passo ({:.2}× o controlo); energia {a:.1} -> {b:.1}", e1[i].1, e1[i].2, loss, loss / base);
    }
}
