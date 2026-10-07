//! Colónias por âncoras numa cena guardada: quantos agentes estão ligados,
//! de que tamanho são os grupos (componentes ligadas), quantas ligações são
//! de nascimento e quantas por contacto, e com que probabilidade de quebra.
//! Só carrega a cena e lê os buffers (não corre passos).
//! SCENE (por omissão o autosave).
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{BOND_STRIDE, Scene, World};

fn find(p: &mut Vec<usize>, mut i: usize) -> usize {
    while p[i] != i {
        p[i] = p[p[i]];
        i = p[i];
    }
    i
}

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let agents = w.read_agents_blocking(&gpu);
    let raw: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.bonds_buf)).to_vec();
    let alive = agents.iter().filter(|a| a.alive != 0).count();
    let mut parent: Vec<usize> = (0..agents.len()).collect();
    let (mut ends, mut birth) = (0u32, 0u32);
    let mut degree: HashMap<u32, u32> = HashMap::new();
    let mut breaks: HashMap<String, u32> = HashMap::new();
    let mut linked = 0u32;
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let mut d = 0;
        // Só as 4 ligações: a 5.ª entrada do slot é a proposta do passo.
        for i in 0..4 {
            let o = (slot * BOND_STRIDE as usize + i) * 4;
            let b = [raw[o], raw[o + 1], raw[o + 2], raw[o + 3]];
            if b[0] == u32::MAX || agents[b[0] as usize].alive == 0 || agents[b[0] as usize].id != b[1] {
                continue;
            }
            d += 1;
            ends += 1;
            birth += (b[2] >> 16 != 0) as u32;
            *breaks.entry(format!("{:.4}", f32::from_bits(b[3]))).or_default() += 1;
            let (x, y) = (find(&mut parent, slot), find(&mut parent, b[0] as usize));
            parent[x] = y;
        }
        let _ = a;
        linked += (d > 0) as u32;
        *degree.entry(d).or_default() += 1;
    }
    let mut size: HashMap<usize, u32> = HashMap::new();
    for slot in (0..agents.len()).filter(|&s| agents[s].alive != 0) {
        *size.entry(find(&mut parent, slot)).or_default() += 1;
    }
    let mut hist: HashMap<u32, u32> = HashMap::new();
    for &n in size.values().filter(|&&n| n > 1) {
        *hist.entry(n).or_default() += 1;
    }
    let mut hist: Vec<_> = hist.into_iter().collect();
    hist.sort();
    let mut degree: Vec<_> = degree.into_iter().collect();
    degree.sort();
    println!("epoch {}, {alive} vivos, {linked} ligados ({:.2}%)", w.params.epoch, 100.0 * linked as f32 / alive.max(1) as f32);
    println!("ligações: {} ({} de nascimento, {} por contacto)", ends / 2, birth / 2, (ends - birth) / 2);
    println!("ligações por agente (n.º de ligações: agentes): {degree:?}");
    println!("grupos (tamanho: quantos): {hist:?}");
    println!("probabilidade de quebra por passo (pontas): {breaks:?}");
    // COMPRIMENTO das ligações (entre os dois resíduos ligados): o repouso é
    // 6 unidades e a ligação parte-se acima de 80. Muitas esticadas = a mola
    // não aguenta o que os agentes e a corrente fazem.
    let pos: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.body_pos_buf)).to_vec();
    let world_of = |slot: usize, k: usize| -> [f32; 2] {
        let a = &agents[slot];
        let (sn, cs) = a.rot.sin_cos();
        let p = pos[slot * 64 + k.min(63)];
        [a.pos_x + cs * p[0] - sn * p[1], a.pos_y + sn * p[0] + cs * p[1]]
    };
    let mut lens: Vec<f32> = Vec::new();
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        for i in 0..4 {
            let o = (slot * BOND_STRIDE as usize + i) * 4;
            let b = [raw[o], raw[o + 1], raw[o + 2]];
            if b[0] == u32::MAX || (b[0] as usize) <= slot || agents[b[0] as usize].alive == 0 || agents[b[0] as usize].id != b[1] {
                continue;
            }
            let _ = a;
            let (p0, p1) = (world_of(slot, (b[2] & 0xFF) as usize), world_of(b[0] as usize, ((b[2] >> 8) & 0xFF) as usize));
            lens.push(((p0[0] - p1[0]).powi(2) + (p0[1] - p1[1]).powi(2)).sqrt());
        }
    }
    if !lens.is_empty() {
        lens.sort_by(f32::total_cmp);
        let q = |f: f32| lens[((lens.len() - 1) as f32 * f) as usize];
        let over = |x: f32| 100.0 * lens.iter().filter(|&&l| l > x).count() as f32 / lens.len() as f32;
        println!(
            "comprimento das ligações (repouso 6, parte a 80): mediana {:.1}, p90 {:.1}, p99 {:.1}, máx {:.1}; acima de 12: {:.1}%, de 30: {:.1}%, de 60: {:.1}%",
            q(0.5), q(0.9), q(0.99), lens[lens.len() - 1], over(12.0), over(30.0), over(60.0)
        );
    }
    // DESTINO das ligações: corre STEPS passos (por omissão 0 = não corre) e
    // vê, das ligações PERMANENTES (probabilidade de quebra 0) que havia,
    // quantas continuam, quantas acabaram por morte de um dos dois e quantas
    // se partiram com os dois vivos (só pode ser por esticarem além de 80).
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if steps > 0 {
        let mut before: Vec<(u32, u32)> = Vec::new();
        for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
            for i in 0..4 {
                let o = (slot * BOND_STRIDE as usize + i) * 4;
                let b = [raw[o], raw[o + 1], raw[o + 2], raw[o + 3]];
                if b[0] == u32::MAX || (b[0] as usize) <= slot || agents[b[0] as usize].alive == 0 || agents[b[0] as usize].id != b[1] {
                    continue;
                }
                if f32::from_bits(b[3]) == 0.0 {
                    before.push((a.id, b[1]));
                }
            }
        }
        let mut done = 0;
        while done < steps {
            let k = ribossome::world::MAX_STEPS_PER_FRAME.min(steps - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += k;
        }
        let after = w.read_agents_blocking(&gpu);
        let raw2: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.bonds_buf)).to_vec();
        let slot_of: HashMap<u32, usize> = after.iter().enumerate().filter(|(_, a)| a.alive != 0).map(|(s, a)| (a.id, s)).collect();
        let (mut kept, mut died, mut snapped) = (0, 0, 0);
        for (ia, ib) in &before {
            match (slot_of.get(ia), slot_of.get(ib)) {
                (Some(&sa), Some(_)) => {
                    let still = (0..4).any(|i| {
                        let o = (sa * BOND_STRIDE as usize + i) * 4;
                        raw2[o] != u32::MAX && raw2[o + 1] == *ib
                    });
                    if still { kept += 1 } else { snapped += 1 }
                }
                _ => died += 1,
            }
        }
        println!(
            "{} ligações permanentes ao fim de {steps} passos: {kept} continuam, {died} acabaram por morte, {snapped} partiram-se com os dois vivos (esticaram além de 80)",
            before.len()
        );
    }
}
