//! A âncora corta os sinais a meio do corpo? Carrega uma cena (qualquer
//! tamanho de mundo), corre STEPS passos (por omissão 300) e compara, por
//! canal, o módulo médio do sinal nos dois resíduos ANTES (lado N) e nos dois
//! DEPOIS (lado C) de um órgão a meio do corpo: âncoras ligadas, âncoras
//! soltas e, como controlo, os outros órgãos (bocas, sensores...). Se a âncora
//! cortasse, o lado a jusante (C para α e γ, N para β e δ) ficava muito abaixo
//! do lado a montante só nas âncoras.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{BOND_STRIDE, MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let gpu = Gpu::new_headless().unwrap();
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let cfg = {
        let c = &scene.header["cfg"];
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        WorldConfig { grid_size: g("grid_size", 2048), fluid_size: g("fluid_size", 1024), world_units_per_cell: g("world_units_per_cell", 30), max_agents: g("max_agents", 400_000) }
    };
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &scene).unwrap();
    let mut done = 0;
    while done < steps {
        let k = (steps - done).min(MAX_STEPS_PER_FRAME);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let agents = w.read_agents_blocking(&gpu);
    let sig: Vec<f32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
    let organs: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
    let raw: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.bonds_buf)).to_vec();
    let s = |slot: usize, k: usize, ch: usize| sig[(slot * 64 + k) * 4 + ch].abs();
    // grupos: 0 âncora ligada, 1 âncora solta, 2 fio, 3 outro órgão, 4 resíduo sem órgão
    let names = ["âncora LIGADA", "âncora solta", "fio (liga dois genes)", "outro órgão (controlo)", "resíduo sem órgão (controlo)"];
    let mut acc = [[[0.0f64; 3]; 4]; 5]; // [grupo][canal][N, aqui, C]
    let mut cnt = [0u32; 5];
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0 && a.body_len >= 7) {
        let n = a.body_len as usize;
        for k in 2..n - 2 {
            let c = organs[slot * 64 + k];
            let group = if c == 0 {
                // (uma amostra de resíduos lisos, para não demorar)
                if (slot + k) % 5 != 0 {
                    continue;
                }
                4
            } else {
                match (c & 0x1F) - 1 {
                    12 => {
                        let bonded = (0..4).any(|i| {
                            let o = (slot * BOND_STRIDE as usize + i) * 4;
                            let t = raw[o] as usize;
                            raw[o] != u32::MAX && t < agents.len() && agents[t].alive != 0 && agents[t].id == raw[o + 1] && (raw[o + 2] & 0xFF) as usize == k
                        });
                        if bonded { 0 } else { 1 }
                    }
                    20 => 2,
                    _ => 3,
                }
            };
            // só conta se os vizinhos não forem eles próprios órgãos (para medir o órgão do meio)
            cnt[group] += 1;
            for ch in 0..4 {
                acc[group][ch][0] += (0.5 * (s(slot, k - 2, ch) + s(slot, k - 1, ch))) as f64;
                acc[group][ch][1] += s(slot, k, ch) as f64;
                acc[group][ch][2] += (0.5 * (s(slot, k + 1, ch) + s(slot, k + 2, ch))) as f64;
            }
        }
    }
    println!("epoch {}, modo de sinal {}, sinais pela ligação {}; módulo médio do sinal (lado N | no órgão | lado C) e passagem a jusante", w.params.epoch, w.params.signal_mode, w.params.bond_signal);
    let ch_names = ["α (vai de N para C)", "β (vai de C para N)", "γ (vai de N para C)", "δ (vai de C para N)"];
    for g in 0..5 {
        if cnt[g] == 0 {
            continue;
        }
        println!("{} ({} casos)", names[g], cnt[g]);
        for ch in 0..4 {
            let v = acc[g][ch].map(|x| x / cnt[g] as f64);
            // a jusante / a montante
            let (up, down) = if ch % 2 == 0 { (v[0], v[2]) } else { (v[2], v[0]) };
            println!("    {:<20} {:.4} | {:.4} | {:.4}   jusante/montante = {:.2}", ch_names[ch], v[0], v[1], v[2], down / up.max(1e-9));
        }
    }
}
