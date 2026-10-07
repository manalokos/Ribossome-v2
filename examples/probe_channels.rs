//! Quem usa que canais? Numa cena (SCENE, por omissão o autosave), depois de
//! STEPS passos: quantos agentes têm α, β, os dois ou nenhum ativos; e, para
//! os órgãos que emitem em α ou β (sensores, relógio, bias), em que canal
//! emitem, onde ficam no corpo (0 = ponta N, 1 = ponta C) e que fração do
//! corpo fica "a jusante" (α corre de N para C, β de C para N: um emissor de
//! α na ponta C, ou de β na ponta N, não chega a ninguém).
use ribossome::gpu::Gpu;
use ribossome::life::organs::ORGAN_NAMES;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let agents = w.read_agents_blocking(&gpu);
    let sig: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
    let ow: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let organ = |slot: usize, k: usize| -> u32 { (ow[slot * 32 + k / 2] >> ((k % 2) * 16)) & 0xFFFF };
    // Tipos que emitem em α ou β pelo "canal" da variante.
    let emits = |t: usize| matches!(t, 2 | 3 | 4 | 5 | 8 | 9 | 13 | 17);
    let mut combo = [0u32; 4]; // nenhum, só α, só β, os dois
    let mut alive = 0u32;
    // Por canal: emissores, soma da posição relativa, soma da fração a jusante, emissores sem ninguém a jusante.
    let mut em = [(0u32, 0.0f64, 0.0f64, 0u32); 2];
    let mut by_type: std::collections::BTreeMap<(usize, usize), u32> = Default::default();
    let mut agents_with = [0u32; 4]; // sem emissor, só emissores α, só β, dos dois
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0 && a.body_len >= 2) {
        alive += 1;
        let n = a.body_len as usize;
        let (mut has_a, mut has_b) = (false, false);
        for k in 0..n {
            has_a |= sig[slot * 64 + k][0].abs() > 0.05;
            has_b |= sig[slot * 64 + k][1].abs() > 0.05;
        }
        combo[(has_a as usize) | ((has_b as usize) << 1)] += 1;
        let (mut ea, mut eb) = (false, false);
        for k in 0..n {
            let o = organ(slot, k);
            if o == 0 {
                continue;
            }
            let (t, p) = (((o & 0x1F) - 1) as usize, ((o >> 5) & 7) as usize);
            if !emits(t) {
                continue;
            }
            let Some(ch) = w.organ_table.get(t).and_then(|r| r.variantes.get(p)).and_then(|v| v.get("canal")) else { continue };
            let ch = (*ch >= 0.5) as usize;
            let rel = k as f64 / (n - 1) as f64;
            let down = if ch == 0 { (n - 1 - k) as f64 } else { k as f64 } / (n - 1) as f64;
            em[ch].0 += 1;
            em[ch].1 += rel;
            em[ch].2 += down;
            em[ch].3 += (down == 0.0) as u32;
            *by_type.entry((t, ch)).or_default() += 1;
            if ch == 0 { ea = true } else { eb = true }
        }
        agents_with[(ea as usize) | ((eb as usize) << 1)] += 1;
    }
    let pct = |v: u32| 100.0 * v as f32 / alive.max(1) as f32;
    println!("epoch {}, {alive} agentes com corpo, modo de sinais {}", w.params.epoch, w.params.signal_mode);
    println!("canais ATIVOS no corpo: nenhum {:.1}%, só α {:.1}%, só β {:.1}%, os dois {:.1}%", pct(combo[0]), pct(combo[1]), pct(combo[2]), pct(combo[3]));
    println!("EMISSORES no corpo: nenhum {:.1}%, só de α {:.1}%, só de β {:.1}%, dos dois {:.1}%", pct(agents_with[0]), pct(agents_with[1]), pct(agents_with[2]), pct(agents_with[3]));
    for (ch, name) in ["α (corre de N para C)", "β (corre de C para N)"].iter().enumerate() {
        let (n, rel, down, dead) = em[ch];
        let m = n.max(1) as f64;
        println!("emissores de {name}: {n}; posição média {:.2} (0 = ponta N, 1 = ponta C); corpo a jusante {:.0}% em média; {:.0}% não têm ninguém a jusante", rel / m, 100.0 * down / m, 100.0 * dead as f64 / m);
    }
    println!("emissores por tipo de órgão (α / β):");
    let mut types: Vec<usize> = by_type.keys().map(|k| k.0).collect();
    types.dedup();
    for t in types {
        println!("  {:38} {:6} / {:6}", ORGAN_NAMES[t], by_type.get(&(t, 0)).copied().unwrap_or(0), by_type.get(&(t, 1)).copied().unwrap_or(0));
    }
}
