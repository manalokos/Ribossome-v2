//! A física deixa um nadador virar para a comida? Agentes DESENHADOS (sem
//! morte nem reprodução) na sopa por omissão (que tem manchas), nos três
//! modos de sinais (0 por aminoácido, 1 isotrópico, 2 direcional, 3
//! direcional com a resposta de cada aminoácido). BODY escolhe o aminoácido
//! do corpo (por omissão G, glicina; S = serina tem a sensibilidade α de
//! sinal contrário):
//! - controlo: relógio + 15 glicinas (o nadador de sempre);
//! - sensor α: sensor de comida direcional (variante 0: canal α, ganho +1) à
//!   cabeça + relógio + 15 glicinas;
//! - sensor β: o mesmo com a variante 1 (canal β): com resposta uniforme deve
//!   virar ao contrário.
//!
//! Mede o índice de viragem (ver probe_chemotaxis): +1 = vira sempre para a
//! comida, 0 = às cegas, −1 = foge.
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

/// Promotor + modificador (da tabela) + GAA (intensidade ×1) para o órgão (tipo, variante).
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

/// O mesmo com o 3.º codão escolhido (no relé é ele que dá os canais).
fn organ3(code: &serde_json::Value, t: u64, v: u64, third: &str) -> String {
    let o = organ(code, t, v);
    format!("{}{third}", &o[..6])
}

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

#[allow(clippy::type_complexity)]
fn main() {
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let code: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("assets/codigo_orgaos.json").unwrap()).unwrap();
    let body_aa = std::env::var("BODY").ok().and_then(|v| v.chars().next()).unwrap_or('G');
    println!("corpo: 15 × {body_aa}");
    let body = codon(body_aa).repeat(15);
    let clock = organ(&code, 5, 0);
    // NOCLOCK=1: os mesmos desenhos SEM relógio (o sensor anda sozinho?):
    // corpo sem órgãos, só sensor α, só sensor β.
    let noclock = std::env::var("NOCLOCK").is_ok();
    // RELAY=1: testa os relés. O sensor α à cabeça; um SWITCH α→γ (3.º codão
    // GGA: entrada α, saída γ, ×2) tira o sinal do canal que dobra o corpo;
    // um segundo SWITCH γ→α (GAG) devolve-o.
    let relay = std::env::var("RELAY").is_ok();
    let sw_ag = organ3(&code, 6, 0, "GGA");
    let sw_ga = organ3(&code, 6, 0, "GAG");
    let sensor = organ(&code, 8, 0);
    let designs = if relay {
        [
            ("sensor α + relógio", format!("AUG{sensor}{clock}{body}UAA")),
            ("sensor, α→γ", format!("AUG{sensor}{sw_ag}{clock}{body}UAA")),
            ("sensor, α→γ, γ→α", format!("AUG{sensor}{sw_ag}GGU{sw_ga}{clock}{body}UAA")),
        ]
    } else if noclock {
        [
            ("corpo sem órgãos", format!("AUG{body}UAA")),
            ("só sensor α", format!("AUG{}{body}UAA", organ(&code, 8, 0))),
            ("só sensor β", format!("AUG{}{body}UAA", organ(&code, 8, 1))),
        ]
    } else {
        [
            ("controlo (relógio)", format!("AUG{clock}{body}UAA")),
            ("sensor α + relógio", format!("AUG{}{clock}{body}UAA", organ(&code, 8, 0))),
            ("sensor β + relógio", format!("AUG{}{clock}{body}UAA", organ(&code, 8, 1))),
        ]
    };
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let g = cfg.grid_size as i32;
    let wpc = cfg.world_units_per_cell as f32;
    let dt = envf("DT").unwrap_or(16.0) as u32;
    let steps = envf("STEPS").unwrap_or(3200.0) as u32;
    let n_each = envf("N").unwrap_or(400.0) as usize;
    // MODES: lista de modos (por omissão todos); ACTIVE: fração ativada da
    // sopa (0,5 dá ~4 ativados por célula; 0,06 dá ~0,5, um mundo pastado).
    let modes: Vec<f32> = std::env::var("MODES")
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|_| vec![0.0, 1.0, 2.0, 3.0]);
    for mode in modes {
        let mut w = World::new(&gpu, cfg, 3);
        let n = cfg.cells() as usize;
        w.custom_terrain = Some((vec![0; n], vec![0.0; n]));
        w.fumaroles.clear();
        w.seed_active = envf("ACTIVE").unwrap_or(0.5);
        w.seed_matter(&gpu, 3);
        {
            let c = w.read_cells_blocking(&gpu);
            let act: u64 = c.iter().map(|v| (v & 0xFFFF) as u64).sum();
            println!("ativados por célula: {:.2}", act as f64 / (c.len() / 4) as f64);
        }
        w.settings.fluid_enabled = false;
        w.settings.terrain_enabled = false;
        w.params.uv_strength = 0.0;
        w.params.settle = 0.0;
        w.params.sedimentation = 0.0;
        w.params.death_probability = 0.0;
        w.params.maintenance_cost = 0.0;
        w.params.motion_cost = 0.0;
        w.params.pairing_rate = 0.0;
        // Por omissão não comem (a sopa e as manchas ficam, e o motor
        // catalítico não mexe as juntas). UPTAKE liga o comer.
        w.params.uptake_rate = envf("UPTAKE").unwrap_or(0.0);
        w.params.spawn_energy = 60.0;
        w.params.diffusion = envf("DIFF").unwrap_or(0.0);
        w.params.aggregation = 0.0;
        w.params.signal_mode = mode;
        let mut rng = ribossome::life::SplitMix(5);
        let s = cfg.sim_size();
        let mut reqs = Vec::new();
        for (_, gnm) in &designs {
            let b = bases(gnm);
            for _ in 0..n_each {
                reqs.push(SpawnRequest::with_genome(s * (0.05 + 0.9 * rng.f32()), s * (0.05 + 0.9 * rng.f32()), &b));
            }
        }
        w.request_seeds(&reqs);
        // KEEP: depois de semear (numa sopa rica, para as sementes nascerem),
        // fica só esta fração dos ativados (o resto passa a gasto): um mundo
        // pastado com os agentes já lá.
        if let Some(keep) = envf("KEEP") {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, 1);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            let mut c = w.read_cells_blocking(&gpu);
            let mut r2 = ribossome::life::SplitMix(99);
            for v in c.iter_mut() {
                let (act, spent) = (*v & 0xFFFF, *v >> 16);
                let k = (0..act).filter(|_| r2.f32() < keep).count() as u32;
                *v = k | ((spent + act - k) << 16);
            }
            gpu.queue.write_buffer(&w.chem_buf, 0, bytemuck::cast_slice(&c));
            let act: u64 = c.iter().map(|v| (v & 0xFFFF) as u64).sum();
            println!("depois de pastar: {:.2} ativados por célula", act as f64 / (c.len() / 4) as f64);
        }
        // Grupo de cada agente pelo comprimento do genoma (os desenhos diferem) e pelo órgão.
        let mut group: HashMap<u32, usize> = HashMap::new();
        let mut turn = [(0f64, 0u64); 3];
        let mut speed = [(0f64, 0u64); 3];
        // Comida à volta (ativados por célula no 5×5) ao longo do percurso.
        let mut food = [(0f64, 0u64); 3];
        let mut prev: HashMap<u32, ([f32; 2], Option<[f32; 2]>, Option<[f32; 2]>)> = HashMap::new();
        let food_dir = |cells: &[u32], x: f32, y: f32| -> Option<[f32; 2]> {
            let (cx, cy) = ((x / wpc) as i32, (y / wpc) as i32);
            let (mut sx, mut sy, mut tot) = (0f32, 0f32, 0f32);
            for dy in -6i32..=6 {
                for dx in -6i32..=6 {
                    if dx * dx + dy * dy > 36 {
                        continue;
                    }
                    let (xx, yy) = (cx + dx, cy + dy);
                    if xx < 0 || yy < 0 || xx >= g || yy >= g {
                        continue;
                    }
                    let i = (yy * g + xx) as usize;
                    let c = (0..4).map(|ch| cells[i * 4 + ch] & 0xFFFF).sum::<u32>() as f32;
                    sx += c * dx as f32;
                    sy += c * dy as f32;
                    tot += c;
                }
            }
            let nrm = (sx * sx + sy * sy).sqrt();
            (tot >= 4.0 && nrm / tot > 0.15).then(|| [sx / nrm, sy / nrm])
        };
        let mut done = 0;
        let mut first = true;
        while done < steps {
            let mut left = dt;
            while left > 0 {
                let k = MAX_STEPS_PER_FRAME.min(left);
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                w.encode_steps(&gpu.queue, &mut enc, k);
                gpu.queue.submit([enc.finish()]);
                gpu.wait_idle();
                left -= k;
            }
            done += dt;
            let cells = w.read_cells_blocking(&gpu);
            let agents = w.read_agents_blocking(&gpu);
            if first {
                first = false;
                let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
                let lens: Vec<u32> = designs.iter().map(|(_, g)| g.len() as u32).collect();
                for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
                    if relay {
                        // Os três desenhos têm comprimentos de genoma diferentes.
                        if let Some(gi) = lens.iter().position(|&l| l == a.gene_len) {
                            group.insert(a.id, gi);
                        }
                        continue;
                    }
                    // 0 controlo; 1 sensor variante 0 (α); 2 sensor variante 1 (β).
                    let mut gi = 0;
                    for r in 0..a.body_len as usize {
                        let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
                        if o != 0 && (o & 0xF) - 1 == 8 {
                            gi = 1 + ((o >> 4) & 0xF) as usize;
                        }
                    }
                    group.insert(a.id, gi.min(2));
                }
            }
            let mut now = HashMap::new();
            for a in agents.iter().filter(|a| a.alive != 0) {
                let Some(&gi) = group.get(&a.id) else { continue };
                let p = [a.pos_x, a.pos_y];
                let mut heading = None;
                if let Some((pp, ph, pf)) = prev.get(&a.id) {
                    let d = [p[0] - pp[0], p[1] - pp[1]];
                    let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                    speed[gi].0 += (len / wpc) as f64;
                    speed[gi].1 += 1;
                    if len > 0.05 * wpc {
                        let h = [d[0] / len, d[1] / len];
                        heading = Some(h);
                        if let (Some(f), Some(h0)) = (pf, ph) {
                            let t = h0[0] * h[1] - h0[1] * h[0];
                            let sd = h0[0] * f[1] - h0[1] * f[0];
                            if t.abs() > 0.02 && sd.abs() > 0.2 {
                                turn[gi].0 += (t.signum() * sd.signum()) as f64;
                                turn[gi].1 += 1;
                            }
                        }
                    }
                }
                {
                    let (cx, cy) = ((p[0] / wpc) as i32, (p[1] / wpc) as i32);
                    let mut sum = 0u32;
                    for dy in -2i32..=2 {
                        for dx in -2i32..=2 {
                            let (xx, yy) = ((cx + dx).clamp(0, g - 1), (cy + dy).clamp(0, g - 1));
                            let i = (yy * g + xx) as usize;
                            sum += (0..4).map(|ch| cells[i * 4 + ch] & 0xFFFF).sum::<u32>();
                        }
                    }
                    // Só a segunda metade da corrida (depois de terem tempo de se orientar).
                    if done > steps / 2 {
                        food[gi].0 += sum as f64 / 25.0;
                        food[gi].1 += 1;
                    }
                }
                now.insert(a.id, (p, heading, food_dir(&cells, p[0], p[1])));
            }
            prev = now;
        }
        println!("modo {mode}:");
        for (gi, (name, _)) in designs.iter().enumerate() {
            let cnt = group.values().filter(|&&x| x == gi).count();
            println!(
                "  {name:22} {cnt:4} agentes: viragem para a comida {:+.3} (n={}), {:.2} células por intervalo, comida à volta {:.3}",
                turn[gi].0 / turn[gi].1.max(1) as f64,
                turn[gi].1,
                speed[gi].0 / speed[gi].1.max(1) as f64,
                food[gi].0 / food[gi].1.max(1) as f64
            );
        }
    }
}
