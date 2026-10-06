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
        for i in 0..BOND_STRIDE as usize {
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
}
