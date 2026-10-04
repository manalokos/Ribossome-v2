//! Osciladores sem relógio: emissão por contacto. Corpos DESENHADOS em arco:
//! arginina (R) à cabeça, N lisinas (K, que enrolam o corpo para um lado) e
//! um aspartato (D) na ponta. Quando a ponta dá a volta e o D toca no R, o
//! R emite β, que (modo de sinais 2) dobra as juntas para o outro lado e
//! abre o anel; sem contacto o sinal apaga-se e o corpo volta a enrolar.
//! Para cada N mede: fração do tempo em contacto, nº de ciclos (ligar ->
//! desligar) e a distância percorrida, contra o controlo (alanina em vez de
//! arginina: não emite). Sem morte, sem comer, sem reprodução.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let wpc = cfg.world_units_per_cell as f32;
    let steps = envf("STEPS", 2000.0) as u32;
    let mode = envf("MODE", 2.0);
    println!("modo dos sinais {mode}; {steps} passos; por desenho: com arginina (emite) | controlo com alanina");
    for n_k in [14usize, 16, 17, 18, 19, 20, 22] {
        let mut w = World::new(&gpu, cfg, 3);
        let n = cfg.cells() as usize;
        w.custom_terrain = Some((vec![0; n], vec![0.0; n]));
        w.fumaroles.clear();
        w.seed_matter(&gpu, 3);
        w.settings.fluid_enabled = false;
        w.settings.terrain_enabled = false;
        w.params.uv_strength = 0.0;
        w.params.settle = 0.0;
        w.params.sedimentation = 0.0;
        w.params.death_probability = 0.0;
        w.params.maintenance_cost = 0.0;
        w.params.motion_cost = 0.0;
        w.params.pairing_rate = 0.0;
        w.params.uptake_rate = 0.0;
        w.params.spawn_energy = 60.0;
        w.params.diffusion = 0.0;
        w.params.aggregation = 0.0;
        w.params.signal_mode = mode;
        let loop_k = "AAA".repeat(n_k);
        let designs = [format!("AUGCGU{loop_k}GAUUAA"), format!("AUGGCU{loop_k}GAUUAA")];
        let mut rng = ribossome::life::SplitMix(5);
        let s = cfg.sim_size();
        let mut reqs = Vec::new();
        for g in &designs {
            let b = bases(g);
            for _ in 0..150 {
                reqs.push(SpawnRequest::with_genome(s * (0.05 + 0.9 * rng.f32()), s * (0.05 + 0.9 * rng.f32()), &b));
            }
        }
        w.request_seeds(&reqs);
        // Por agente: grupo (0 emissor, 1 controlo), posição anterior, percurso,
        // contacto anterior, passos em contacto, ciclos.
        struct T {
            group: usize,
            pos: [f32; 2],
            path: f32,
            on: bool,
            on_steps: u32,
            cycles: u32,
            samples: u32,
        }
        let mut track: HashMap<u32, T> = HashMap::new();
        let mut done = 0;
        while done < steps {
            let k = 4.min(MAX_STEPS_PER_FRAME).min(steps - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += k;
            let agents = w.read_agents_blocking(&gpu);
            let sig: Vec<f32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
            let organs: Vec<u8> = gpu.read_buffer_blocking(&w.bodies_buf);
            for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
                // Resíduo 1 (a seguir ao M): arginina (índice 14 em AMINO) = emissor.
                let aa1 = organs[slot * 64 + 1];
                // β (canal 1) no resíduo 1: > 0,5 = está a emitir (em contacto).
                let on = sig[(slot * 64 + 1) * 4 + 1] > 0.5;
                let t = track.entry(a.id).or_insert(T {
                    group: if aa1 == 14 { 0 } else { 1 },
                    pos: [a.pos_x, a.pos_y],
                    path: 0.0,
                    on,
                    on_steps: 0,
                    cycles: 0,
                    samples: 0,
                });
                t.path += ((a.pos_x - t.pos[0]).powi(2) + (a.pos_y - t.pos[1]).powi(2)).sqrt();
                t.pos = [a.pos_x, a.pos_y];
                if t.on && !on {
                    t.cycles += 1;
                }
                t.on = on;
                t.on_steps += on as u32;
                t.samples += 1;
            }
        }
        let mut line = format!("  {n_k:2} lisinas:");
        for gi in 0..2 {
            let v: Vec<&T> = track.values().filter(|t| t.group == gi).collect();
            let m = v.len().max(1) as f32;
            line += &format!(
                "  {} {:3} ag., contacto {:4.1}% do tempo, {:5.1} ciclos, percurso {:6.1} células |",
                if gi == 0 { "R:" } else { "controlo:" },
                v.len(),
                100.0 * v.iter().map(|t| t.on_steps as f32 / t.samples.max(1) as f32).sum::<f32>() / m,
                v.iter().map(|t| t.cycles as f32).sum::<f32>() / m,
                v.iter().map(|t| t.path).sum::<f32>() / m / wpc
            );
        }
        println!("{line}");
    }
}
