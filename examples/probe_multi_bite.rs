//! Um predador a morder várias presas ao mesmo tempo recebe mesmo a energia
//! de todas? Mundo de teste vazio (sem comida, sem morte natural, sem
//! manutenção, sem movimento): um predador com uma protease de contacto
//! sempre ligada no meio e N presas (por omissão 5) à volta, a tocar-lhe.
//! Em cada passo mostra o que o predador tirou (contact_disp.z), a soma do
//! que as presas perderam (contact_disp.x) e as energias. Tem de dar:
//! tirado = soma das perdas, e a energia do predador a subir
//! protease_direct × tirado por passo (até à capacidade dele).
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
    let prey_n = envf("PREY", 5.0) as usize;
    let radius = envf("RADIUS", 9.0);
    let code: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/codigo_orgaos.json").unwrap()).unwrap();
    // Predador: protease de contacto sempre ligada + depósitos (para caber a energia que tira).
    let pred = bases(&format!("AUG{}{}{}UAA", organ(&code, 11, 0), organ(&code, 7, 0), organ(&code, 7, 0)));
    let prey = bases(&format!("AUG{}UAA", codon('G').repeat(4)));
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
    w.params.naked_decay = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.motion_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.brownian = 0.0;
    w.params.brownian_rot = 0.0;
    w.params.store_loss = 0.0;
    w.params.spawn_energy = envf("ENERGY", 1.0);
    w.params.protease_power = envf("POWER", 1.0);
    w.params.protease_direct = envf("DIRECT", 0.8);
    let c = cfg.sim_size() * 0.5;
    let mut reqs = vec![SpawnRequest::with_genome(c, c, &pred)];
    for i in 0..prey_n {
        let a = i as f32 / prey_n as f32 * std::f32::consts::TAU;
        reqs.push(SpawnRequest::with_genome(c + radius * a.cos(), c + radius * a.sin(), &prey));
    }
    w.request_seeds(&reqs);
    println!("1 predador (protease de contacto) e {prey_n} presas a {radius} unidades; força das proteases {}, direta {}, energia inicial {}", w.params.protease_power, w.params.protease_direct, w.params.spawn_energy);
    println!("passo | predador: tirou   energia | presas: soma das perdas   mordidas   energia de cada uma");
    for step in 1..=envf("STEPS", 10.0) as u32 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, 1);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        let agents = w.read_agents_blocking(&gpu);
        let bite: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.contact_disp_buf)).to_vec();
        let alive: Vec<usize> = (0..agents.len()).filter(|&s| agents[s].alive != 0).collect();
        let p = alive.iter().copied().find(|&s| agents[s].gene_len as usize == pred.len());
        let preys: Vec<usize> = alive.iter().copied().filter(|&s| agents[s].gene_len as usize == prey.len()).collect();
        let lost: f32 = preys.iter().map(|&s| bite[s][0]).sum();
        let bitten = preys.iter().filter(|&&s| bite[s][0] > 0.0).count();
        let es: Vec<String> = preys.iter().map(|&s| format!("{:.2}", agents[s].energy)).collect();
        match p {
            Some(p) => println!("{step:>5} | {:>15.3} {:>9.3} | {:>22.3} {:>10} de {}   {}", bite[p][2], agents[p].energy, lost, bitten, preys.len(), es.join(" ")),
            None => println!("{step:>5} | o predador morreu"),
        }
    }
}
