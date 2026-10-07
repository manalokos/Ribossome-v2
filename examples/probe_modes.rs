//! Quanto nadam corpos AO ACASO em cada modo de sinais? As mesmas N sementes
//! (como o botão "semear"), sem fluido, contacto, browniano, morte nem
//! nascimentos: só a natação pela mudança de forma. Mede o deslocamento de
//! cada agente em STEPS passos (em células) e compara:
//!   modo 2 (condução N->C e resposta uniformes),
//!   modo 0 (condução e resposta da tabela),
//!   modo 4 (condução da tabela, resposta uniforme),
//!   modo 3 (condução uniforme, resposta da tabela),
//!   modo 0 com a resposta da tabela × SENS (por omissão 0,5).
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let n = envf("N", 4000.0) as u32;
    let steps = envf("STEPS", 3000.0) as u32;
    let sens = envf("SENS", 0.5);
    // (nome, modo, × resposta, variante da tabela)
    // variantes: 1 = β conduz como α (N->C); 2 = sem isolantes nem
    // inversores; 3 = as duas; 4 = sem exceções de sinal na resposta;
    // 5 = só F isola, só P inverte e só D responde ao contrário; 6 = nenhuma
    // exceção (condução e resposta com os valores de cada um, mas coerentes);
    // 10 = sem resíduo nos outros canais.
    let cases: [(&str, f32, f32, u32); 12] = [
        ("modo 2 (tudo uniforme)", 2.0, 1.0, 0),
        ("modo 0 (tabela)", 0.0, 1.0, 0),
        ("modo 4 (condução da tabela)", 4.0, 1.0, 0),
        ("modo 4, β também N->C", 4.0, 1.0, 1),
        ("modo 4, sem isol./invers.", 4.0, 1.0, 2),
        ("modo 4, as duas coisas", 4.0, 1.0, 3),
        ("modo 3 (resposta da tabela)", 3.0, 1.0, 0),
        ("modo 3, sem exceções de sinal", 3.0, 1.0, 4),
        ("modo 0, resposta × SENS", 0.0, sens, 0),
        ("modo 0, 1 isol. 1 inv. 1 exc.", 0.0, 1.0, 5),
        ("modo 0, sem exceções nenhumas", 0.0, 1.0, 6),
        ("modo 0, sem resíduo", 0.0, 1.0, 10),
    ];
    println!("{n} sementes, {steps} passos; deslocamento em células");
    let only: Option<Vec<usize>> = std::env::var("CASES").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
    for (ci, (name, mode, k, variant)) in cases.into_iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(&ci)) {
            continue;
        }
        let mut w = World::new(&gpu, cfg, 7);
        w.seed_matter(&gpu, 7);
        w.settings.fluid_enabled = false;
        w.settings.contact_enabled = false;
        w.params.brownian = 0.0;
        w.params.brownian_rot = 0.0;
        w.params.phoretic_gain = 0.0;
        w.params.pairing_rate = 0.0;
        w.params.death_probability = 0.0;
        w.params.maintenance_cost = 0.0;
        w.params.motion_cost = 0.0;
        w.params.uv_strength = 0.0;
        w.params.sedimentation = 0.0;
        w.params.spawn_energy = 50.0;
        w.params.signal_mode = mode;
        if variant == 10 {
            w.params.signal_crosstalk = 0.0;
        }
        if k != 1.0 || (variant != 0 && variant < 10) {
            let mut rows = w.amino.clone();
            for r in rows.iter_mut() {
                r.sens_alfa *= k;
                r.sens_beta *= k;
                if variant == 2 || variant == 3 {
                    // Isolantes (0 / 0) e inversores (negativos) passam a conduzir como os outros.
                    if r.cond_alfa_n <= 0.0 {
                        (r.cond_alfa_n, r.cond_alfa_c, r.cond_beta_n, r.cond_beta_c) = (0.95, 0.05, 0.05, 0.95);
                    }
                }
                if variant == 1 || variant == 3 {
                    (r.cond_beta_n, r.cond_beta_c) = (r.cond_alfa_n, r.cond_alfa_c);
                }
                let l = r.letra.as_str();
                if variant == 5 || variant == 6 {
                    let keep = variant == 5 && (l == "F" || l == "P");
                    if r.cond_alfa_n <= 0.0 && !keep {
                        (r.cond_alfa_n, r.cond_alfa_c, r.cond_beta_n, r.cond_beta_c) = (0.95, 0.05, 0.05, 0.95);
                    }
                    let flip = variant == 5 && l == "D";
                    r.sens_alfa = if flip { -r.sens_alfa.abs() } else { r.sens_alfa.abs() };
                    r.sens_beta = if flip { r.sens_beta.abs() } else { -r.sens_beta.abs() };
                }
                if variant == 4 {
                    r.sens_alfa = r.sens_alfa.abs();
                    r.sens_beta = -r.sens_beta.abs();
                }
            }
            w.set_amino(&gpu.queue, rows);
        }
        let mut rng = ribossome::life::SplitMix(11);
        let reqs: Vec<SpawnRequest> = ribossome::life::seed_requests(n, [12, 120], true, cfg.sim_size(), &mut rng);
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
        run(&mut w, 200);
        let a0: HashMap<u32, (f32, f32)> =
            w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0 && a.body_len >= 2).map(|a| (a.id, (a.pos_x, a.pos_y))).collect();
        run(&mut w, steps);
        let wpc = cfg.world_units_per_cell as f32;
        let mut d: Vec<f32> = w
            .read_agents_blocking(&gpu)
            .iter()
            .filter(|a| a.alive != 0)
            .filter_map(|a| a0.get(&a.id).map(|p| ((a.pos_x - p.0).powi(2) + (a.pos_y - p.1).powi(2)).sqrt() / wpc))
            .collect();
        d.sort_by(f32::total_cmp);
        let q = |f: f32| d[((d.len() - 1) as f32 * f) as usize];
        let moved = |c: f32| 100.0 * d.iter().filter(|&&x| x > c).count() as f32 / d.len() as f32;
        println!(
            "{name:30} {} corpos | mediana {:.2}, p90 {:.2}, p99 {:.1}, máx {:.1} | > 1 célula: {:.1}%, > 5: {:.1}%, > 20: {:.2}%",
            d.len(),
            q(0.5),
            q(0.9),
            q(0.99),
            d[d.len() - 1],
            moved(1.0),
            moved(5.0),
            moved(20.0)
        );
    }
}
