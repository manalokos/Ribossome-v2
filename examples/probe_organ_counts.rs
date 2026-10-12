//! Quantos órgãos do mesmo tipo tem cada agente? Carrega uma cena (qualquer
//! tamanho de mundo) e, para o tipo ORGAN (por omissão 10 = fotossistema),
//! mostra quantos agentes têm 0, 1, 2, 3... desses órgãos, com a intensidade
//! média, o tamanho do corpo e a energia de cada grupo, e alguns exemplos
//! (a sequência de órgãos do corpo) dos que têm mais de um.
use std::collections::BTreeMap;

use ribossome::gpu::Gpu;
use ribossome::life::organs::{ORGAN_NAMES_EN, ORGAN_SYMBOLS};
use ribossome::params::WorldConfig;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let kind: u16 = std::env::var("ORGAN").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let gpu = Gpu::new_headless().unwrap();
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let cfg = {
        let c = &scene.header["cfg"];
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        WorldConfig { grid_size: g("grid_size", 2048), fluid_size: g("fluid_size", 1024), world_units_per_cell: g("world_units_per_cell", 30), max_agents: g("max_agents", 400_000) }
    };
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &scene).unwrap();
    let agents = w.read_agents_blocking(&gpu);
    let organs: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
    // nº de órgãos do tipo -> (agentes, soma das intensidades, soma dos resíduos, soma da energia)
    let mut by: BTreeMap<usize, (u32, f64, f64, f64)> = BTreeMap::new();
    let mut examples: BTreeMap<String, u32> = BTreeMap::new();
    let mut alive = 0u32;
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        alive += 1;
        let body = &organs[slot * 64..slot * 64 + (a.body_len as usize).min(64)];
        let mine: Vec<f32> = body.iter().filter(|&&c| c != 0 && (c & 0x1F) - 1 == kind).map(|&c| 2f32.powf(((c >> 8) as f32 - 32.0) / 8.0)).collect();
        let e = by.entry(mine.len()).or_default();
        e.0 += 1;
        e.1 += mine.iter().sum::<f32>() as f64;
        e.2 += a.body_len as f64;
        e.3 += a.energy as f64;
        if mine.len() >= 2 {
            let s: String = body.iter().filter(|&&c| c != 0).map(|&c| ORGAN_SYMBOLS[((c & 0x1F) - 1) as usize]).collect();
            *examples.entry(format!("{s} ({} resíduos)", a.body_len)).or_default() += 1;
        }
    }
    println!("epoch {}: {alive} vivos; órgão {kind} = {} ({})", w.params.epoch, ORGAN_NAMES_EN[kind as usize], ORGAN_SYMBOLS[kind as usize]);
    for (n, (c, gain, res, en)) in &by {
        let k = *c as f64;
        let per = if *n > 0 { format!(", intensidade média ×{:.2} por órgão", gain / (k * *n as f64)) } else { String::new() };
        println!("  com {n}: {c:>7} agentes ({:>5.2}%), {:.1} resíduos, energia {:.2}{per}", 100.0 * k / alive.max(1) as f64, res / k, en / k);
    }
    let mut ex: Vec<(String, u32)> = examples.into_iter().collect();
    ex.sort_by(|a, b| b.1.cmp(&a.1));
    if !ex.is_empty() {
        println!("corpos com mais de um (os mais comuns):");
        for (s, c) in ex.iter().take(12) {
            println!("  {c:>6}  {s}");
        }
    }
}
