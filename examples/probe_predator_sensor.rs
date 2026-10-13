//! O sensor de predadores reage a uma protease aberta ao lado? Mundo de teste
//! vazio: um corpo com o sensor (variante 0, emite em α) e, a DIST unidades,
//! um corpo com uma protease de contacto sempre aberta; depois o mesmo com um
//! corpo sem protease no lugar dele (o controlo). Mostra o sinal α no resíduo
//! do sensor ao longo de alguns passos.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

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
    let dist = envf("DIST", 40.0);
    let code: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/codigo_orgaos.json").unwrap()).unwrap();
    let g4 = codon('G').repeat(4);
    let sensor = bases(&format!("AUG{}{g4}UAA", organ(&code, 1, 0)));
    let armed = bases(&format!("AUG{}{g4}UAA", organ(&code, 11, 0)));
    let plain = bases(&format!("AUG{}UAA", codon('G').repeat(7)));
    for (name, other) in [("vizinho COM protease aberta", &armed), ("vizinho sem protease (controlo)", &plain)] {
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
        w.params.brownian = 0.0;
        w.params.brownian_rot = 0.0;
        w.params.protease_power = 0.0;
        w.params.spawn_energy = 2.0;
        let c = cfg.sim_size() * 0.5;
        w.request_seeds(&[SpawnRequest::with_genome(c, c, &sensor), SpawnRequest::with_genome(c + dist, c, other)]);
        let mut line = String::new();
        for _ in 0..12 {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, 5);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            let agents = w.read_agents_blocking(&gpu);
            let sig: Vec<f32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
            let org: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
            for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
                for k in 0..a.body_len as usize {
                    let o = org[slot * 64 + k];
                    if o != 0 && (o & 0x1F) - 1 == 1 {
                        line += &format!(" {:+.2}", sig[(slot * 64 + k) * 4]);
                    }
                }
            }
        }
        println!("{name}, a {dist} unidades: sinal α no sensor de 5 em 5 passos:{line}");
    }
}
