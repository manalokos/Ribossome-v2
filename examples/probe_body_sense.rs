//! O sensor de corpos distingue presas pela composição? Sensores desenhados
//! (sensor de comida total, variante "corpos", emite em β com ganho −2) com
//! quatro antenas diferentes (o resíduo a seguir ao sensor: E, K, L, G), cada
//! um no meio de um anel de presas: ricas em lisina (o que a família 1 corta),
//! em aspartato (família 2) ou em leucina (família 3). Lê o sinal β no sensor.
//! Espera-se: antena E só reage às de lisina, K às de aspartato, L às de
//! leucina, e G a todas.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let mut w = World::new(&gpu, cfg, 3);
    w.custom_terrain = Some((vec![0; cfg.cells() as usize], vec![0.0; cfg.cells() as usize]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.settings.contact_enabled = false;
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.brownian_rot = 0.0;
    w.params.uv_strength = 0.0;
    w.params.sedimentation = 0.0;
    w.params.spawn_energy = 50.0;
    // Q (CAA) + D (GAU) = sensor de comida, variante 3 (alvo: corpos); GAA = intensidade.
    let antennas = [("E (ácido)", "GAA"), ("K (básico)", "AAA"), ("L (hidrofóbico)", "CUU"), ("G (neutro)", "GGU")];
    let prey = [("lisina", "AAA"), ("aspartato", "GAU"), ("leucina", "CUU")];
    let s = cfg.sim_size();
    let mut reqs = Vec::new();
    let mut sensors: HashMap<Vec<u8>, usize> = HashMap::new();
    let per = 12;
    for (ai, (_, codon)) in antennas.iter().enumerate() {
        let g = bases(&format!("AUG CAA GAU GAA {codon} GGU GGU GGU UAA"));
        sensors.insert(g.clone(), ai);
        for (pi, (_, pc)) in prey.iter().enumerate() {
            let pg = bases(&format!("AUG {} UAA", format!("{pc} ").repeat(10)));
            for r in 0..per {
                // Uma grelha de postos bem separados; a zona (terço do mundo) diz a presa.
                let x = s * (0.05 + 0.9 * (ai * per + r) as f32 / (antennas.len() * per) as f32);
                let y = s * (0.2 + 0.3 * pi as f32);
                reqs.push(SpawnRequest::with_genome(x, y, &g));
                for q in 0..6 {
                    let a = q as f32 * std::f32::consts::TAU / 6.0;
                    reqs.push(SpawnRequest::with_genome(x + 55.0 * a.cos(), y + 55.0 * a.sin(), &pg));
                }
            }
        }
    }
    w.request_seeds(&reqs);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    w.encode_steps(&gpu.queue, &mut enc, 40);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let agents = w.read_agents_blocking(&gpu);
    let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
    let sig: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
    let mut sum = vec![[(0.0f32, 0u32); 3]; antennas.len()];
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let g: Vec<u8> = (0..a.gene_len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
        if let Some(&ai) = sensors.get(&g) {
            let pi = (((a.pos_y / s - 0.2) / 0.3).round() as usize).min(2);
            // O sensor é o resíduo 1; emite em β.
            sum[ai][pi].0 += sig[slot * 64 + 1][1];
            sum[ai][pi].1 += 1;
        }
    }
    println!("sinal β médio no sensor (ganho −2: mais negativo = mais corpos sentidos)");
    println!("{:18} {:>10} {:>10} {:>10}", "antena", prey[0].0, prey[1].0, prey[2].0);
    for (ai, (name, _)) in antennas.iter().enumerate() {
        let v: Vec<String> = sum[ai].iter().map(|(t, n)| format!("{:10.3}", t / (*n).max(1) as f32)).collect();
        println!("{name:18} {}", v.join(" "));
    }
}
