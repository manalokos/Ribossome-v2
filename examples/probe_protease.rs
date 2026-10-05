//! Proteases por contacto, digestão e bias de idade, com corpos DESENHADOS
//! (mundo sem comida, sem morte, sem manutenção; toda a matéria GASTA à
//! partida, para se ver a digestão a reativá-la):
//! - predador: serina + N lisinas (que enrolam o corpo em anel) + histidina
//!   na ponta: quando o anel fecha, S toca em H = sítio de protease de
//!   serina (corta lisina e arginina). O próprio predador é feito de lisina,
//!   por isso os predadores mordem-se uns aos outros (a regra é cega);
//! - controlo do predador: alanina em vez de serina (sem sítio);
//! - presa K: 12 lisinas (alvo); presa G: 12 glicinas (sem alvo: imune);
//! - bias de idade: órgão 11 variante 0 (α, +1, meia-vida 300) + 12 glicinas.
//! Mede a energia perdida por desenho, as mordidas, os monómeros reativados
//! (≈ energia tirada / energia por monómero), o sinal δ nos predadores que
//! estão a morder e o sinal α do bias de idade ao longo da vida.
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

fn organ(code: &serde_json::Value, t: u64, v: u64) -> String {
    for (p, mods) in code.as_object().unwrap() {
        for (m, tv) in mods.as_object().unwrap() {
            if tv[0].as_u64() == Some(t) && tv[1].as_u64() == Some(v) {
                return format!("{}{}GAA", codon(p.chars().next().unwrap()), codon(m.chars().next().unwrap()));
            }
        }
    }
    panic!("órgão {t}/{v} não está na tabela");
}

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let code: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/codigo_orgaos.json").unwrap()).unwrap();
    let ring = envf("RING", 19.0) as usize;
    let k_loop = codon('K').repeat(ring);
    let designs = [
        ("predador (S + lisinas + H)", format!("AUG{}{k_loop}{}UAA", codon('S'), codon('H'))),
        ("controlo (A + lisinas + H)", format!("AUG{}{k_loop}{}UAA", codon('A'), codon('H'))),
        ("presa K (12 lisinas)", format!("AUG{}UAA", codon('K').repeat(12))),
        ("presa G (12 glicinas)", format!("AUG{}UAA", codon('G').repeat(12))),
        ("bias de idade + 12 glicinas", format!("AUG{}{}UAA", organ(&code, 11, 0), codon('G').repeat(12))),
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
    w.params.maintenance_cost = 0.0;
    w.params.motion_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.spawn_energy = 60.0;
    w.params.signal_mode = 2.0;
    w.params.brownian = envf("BROWNIAN", 10.0);
    w.params.protease_power = envf("POWER", 1.0);
    let mut rng = ribossome::life::SplitMix(5);
    let s = cfg.sim_size();
    let per = envf("PER", 250.0) as usize;
    // Todos juntos num quarto do mundo, para haver contactos.
    let mut reqs = Vec::new();
    let mut by_genome: HashMap<Vec<u8>, usize> = HashMap::new();
    for (i, (_, g)) in designs.iter().enumerate() {
        let b = bases(g);
        by_genome.insert(b.clone(), i);
        for _ in 0..per {
            reqs.push(SpawnRequest::with_genome(s * (0.3 + 0.4 * rng.f32()), s * (0.3 + 0.4 * rng.f32()), &b));
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
    // (energia média, agentes, sinal α médio no resíduo 1, fração a morder, δ médio no resíduo 1 de quem morde)
    let sample = |w: &World| -> Vec<(f32, u32, f32, f32, f32)> {
        let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
        let sig: Vec<f32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
        let bite: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.contact_disp_buf)).to_vec();
        let mut out = vec![(0f32, 0u32, 0f32, 0f32, 0f32); designs.len()];
        let mut biters = vec![0u32; designs.len()];
        for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
            let g: Vec<u8> = (0..a.gene_len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
            if let Some(&i) = by_genome.get(&g) {
                out[i].0 += a.energy;
                out[i].1 += 1;
                out[i].2 += sig[(slot * 64 + 1) * 4];
                if bite[slot][2] > 0.0 {
                    biters[i] += 1;
                    out[i].4 += sig[(slot * 64 + 1) * 4 + 3];
                }
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            let m = o.1.max(1) as f32;
            o.0 /= m;
            o.2 /= m;
            o.3 = biters[i] as f32 / m;
            o.4 /= biters[i].max(1) as f32;
        }
        out
    };
    run(&mut w, 30);
    let l0 = w.ledger_blocking(&gpu);
    let c0 = w.life_counters_blocking(&gpu);
    let e0 = sample(&w);
    println!("anel de {ring} lisinas, força ×{}; idade 30: sinal α do bias de idade = {:+.2}", w.params.protease_power, e0[4].2);
    let steps = envf("STEPS", 3000.0) as u32;
    let mut biting_frac = vec![0f32; designs.len()];
    let mut delta = vec![(0f32, 0u32); designs.len()];
    let chunks = 30;
    for c in 0..chunks {
        run(&mut w, steps / chunks);
        let smp = sample(&w);
        for i in 0..designs.len() {
            biting_frac[i] += smp[i].3 / chunks as f32;
            if smp[i].3 > 0.0 {
                delta[i].0 += smp[i].4;
                delta[i].1 += 1;
            }
        }
        if c == 2 || c == 8 {
            println!("idade {}: sinal α do bias de idade = {:+.2}", 30 + (c + 1) * (steps / chunks), smp[4].2);
        }
    }
    let e1 = sample(&w);
    let l1 = w.ledger_blocking(&gpu);
    let c1 = w.life_counters_blocking(&gpu);
    let mut lost_total = 0.0;
    for (i, (name, _)) in designs.iter().enumerate() {
        let lost = (e0[i].0 - e1[i].0) * e1[i].1 as f32;
        lost_total += lost;
        println!(
            "  {name:30} {:3} agentes: energia {:5.2} -> {:5.2}; a morder {:4.1}% do tempo; sinal δ quando morde {:.2}",
            e1[i].1,
            e0[i].0,
            e1[i].0,
            100.0 * biting_frac[i],
            delta[i].0 / delta[i].1.max(1) as f32
        );
    }
    let act = |l: &ribossome::world::Ledger| l.act.iter().map(|&x| x as u64).sum::<u64>();
    println!(
        "mordidas {}; energia perdida no total {:.1}; monómeros reativados {} (esperado ~{:.0} = energia / {}); matéria {} -> {}",
        c1.bites.wrapping_sub(c0.bites),
        lost_total,
        act(&l1) as i64 - act(&l0) as i64,
        lost_total / w.params.food_power,
        w.params.food_power,
        l0.total(),
        l1.total()
    );
}
