//! Colónias por âncoras: vivem como um corpo só? Carrega uma cena (por
//! omissão o autosave; qualquer tamanho de mundo) e junta os agentes ligados
//! por âncoras em colónias. Mostra: quantos estão ligados e em colónias de que
//! tamanho; que PAPÉIS há em cada colónia (produtor = fotossistema ou
//! quimiossíntese, consumidor = boca, caçador = protease) e quantas misturam
//! produtores com consumidores; a energia dos ligados contra a dos soltos, por
//! papel; e quão igual é a energia dentro de uma colónia (a partilha pelas
//! ligações) comparada com a de agentes soltos tirados ao acaso.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{BOND_STRIDE, Scene, World};

const ROLES: [&str; 5] = ["sem captação", "consumidor (boca)", "produtor (luz/quimio)", "produtor com boca", "caçador (protease)"];

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
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
    let raw: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.bonds_buf)).to_vec();
    let organs: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
    let n = agents.len();
    let alive: Vec<usize> = (0..n).filter(|&s| agents[s].alive != 0).collect();
    // Papel de cada agente, pelos órgãos do corpo que tem.
    let role = |s: usize| -> usize {
        let (mut mouth, mut prod, mut hunt) = (false, false, false);
        for &c in &organs[s * 64..s * 64 + (agents[s].body_len as usize).min(64)] {
            if c == 0 {
                continue;
            }
            match (c & 0x1F) - 1 {
                0 => mouth = true,
                10 | 14 => prod = true,
                11 => hunt = true,
                _ => {}
            }
        }
        match (hunt, prod, mouth) {
            (true, _, _) => 4,
            (_, true, true) => 3,
            (_, true, false) => 2,
            (_, false, true) => 1,
            _ => 0,
        }
    };
    let roles: Vec<usize> = (0..n).map(|s| if agents[s].alive != 0 { role(s) } else { 0 }).collect();
    // Ligações válidas -> colónias (união-procura).
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let (mut links, mut birth_links, mut cross_links) = (0u32, 0u32, 0u32);
    for &s in &alive {
        for i in 0..4 {
            let o = (s * BOND_STRIDE as usize + i) * 4;
            let (t, id, z) = (raw[o], raw[o + 1], raw[o + 2]);
            if t == u32::MAX || t as usize >= n || agents[t as usize].alive == 0 || agents[t as usize].id != id {
                continue;
            }
            if (t as usize) > s {
                links += 1;
                birth_links += (z >> 16 != 0) as u32;
                cross_links += (roles[s] != roles[t as usize]) as u32;
            }
            let (a, b) = (find(&mut parent, s as u32), find(&mut parent, t));
            if a != b {
                parent[a as usize] = b;
            }
        }
    }
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    for &s in &alive {
        let r = find(&mut parent, s as u32);
        groups.entry(r).or_default().push(s);
    }
    let colonies: Vec<&Vec<usize>> = groups.values().filter(|g| g.len() >= 2).collect();
    let bonded: usize = colonies.iter().map(|g| g.len()).sum();
    println!("epoch {}: {} vivos, {} ligados ({:.1}%) em {} colónias; {} ligações ({:.0}% de nascença, {:.0}% entre papéis diferentes)", w.params.epoch, alive.len(), bonded, 100.0 * bonded as f32 / alive.len().max(1) as f32, colonies.len(), links, 100.0 * birth_links as f32 / links.max(1) as f32, 100.0 * cross_links as f32 / links.max(1) as f32);
    // Tamanhos.
    let bins = [(2, 2), (3, 3), (4, 5), (6, 10), (11, 30), (31, 100000)];
    println!("tamanho das colónias (agentes): quantas / que parte dos ligados");
    for (lo, hi) in bins {
        let g: Vec<_> = colonies.iter().filter(|g| g.len() >= lo && g.len() <= hi).collect();
        let m: usize = g.iter().map(|g| g.len()).sum();
        if !g.is_empty() {
            println!("  {:>5}{}: {:>6} colónias, {:>5.1}% dos ligados", lo, if hi == lo { "     ".to_string() } else if hi > 1000 { " ou + ".to_string() } else { format!(" a {hi:<2}") }, g.len(), 100.0 * m as f32 / bonded.max(1) as f32);
        }
    }
    println!("  a maior tem {} agentes", colonies.iter().map(|g| g.len()).max().unwrap_or(0));
    // Papéis dentro das colónias.
    let has = |g: &Vec<usize>, f: &dyn Fn(usize) -> bool| g.iter().any(|&s| f(roles[s]));
    let mixed = colonies.iter().filter(|g| has(g, &|r| r == 2 || r == 3) && has(g, &|r| r == 1 || r == 3)).count();
    let split = colonies.iter().filter(|g| has(g, &|r| r == 2) && has(g, &|r| r == 1)).count();
    let single = colonies.iter().filter(|g| g.iter().all(|&s| roles[s] == roles[g[0]])).count();
    let with_idle = colonies.iter().filter(|g| has(g, &|r| r == 0)).count();
    let with_hunter = colonies.iter().filter(|g| has(g, &|r| r == 4)).count();
    let pct = |k: usize| 100.0 * k as f32 / colonies.len().max(1) as f32;
    println!("papéis nas colónias:");
    println!("  {:>5.1}% têm produção E boca (em qualquer membro)", pct(mixed));
    println!("  {:>5.1}% têm DIVISÃO DE TRABALHO: um produtor sem boca ligado a um consumidor sem produção", pct(split));
    println!("  {:>5.1}% são de um só papel", pct(single));
    println!("  {:>5.1}% têm um membro sem captação nenhuma (vive do que os outros lhe passam)", pct(with_idle));
    println!("  {:>5.1}% têm um caçador", pct(with_hunter));
    // Energia: ligados contra soltos, por papel (energia e energia por resíduo).
    let is_bonded: Vec<bool> = {
        let mut v = vec![false; n];
        for g in &colonies {
            for &s in g.iter() {
                v[s] = true;
            }
        }
        v
    };
    println!("energia média (e por resíduo) por papel: soltos | ligados");
    for (r, name) in ROLES.iter().enumerate() {
        let stat = |b: bool| {
            let v: Vec<&ribossome::params::Agent> = alive.iter().filter(|&&s| roles[s] == r && is_bonded[s] == b && agents[s].body_len > 0).map(|&s| &agents[s]).collect();
            let k = v.len().max(1) as f32;
            (v.len(), v.iter().map(|a| a.energy).sum::<f32>() / k, v.iter().map(|a| a.energy / a.body_len as f32).sum::<f32>() / k)
        };
        let (f, l) = (stat(false), stat(true));
        if f.0 + l.0 > 0 {
            println!("  {name:<24} soltos {:>6}: {:>5.2} ({:.3}/res) | ligados {:>6}: {:>5.2} ({:.3}/res)", f.0, f.1, f.2, l.0, l.1, l.2);
        }
    }
    // Partilha: quão diferente é a energia por resíduo entre dois membros
    // ligados, contra dois agentes soltos quaisquer (diferença absoluta média).
    let per = |s: usize| agents[s].energy / agents[s].body_len.max(1) as f32;
    let mut d_link = Vec::new();
    for &s in &alive {
        for i in 0..4 {
            let o = (s * BOND_STRIDE as usize + i) * 4;
            let t = raw[o] as usize;
            if raw[o] != u32::MAX && t < n && t > s && agents[t].alive != 0 && agents[t].id == raw[o + 1] {
                d_link.push((per(s) - per(t)).abs());
            }
        }
    }
    let free: Vec<usize> = alive.iter().copied().filter(|&s| !is_bonded[s] && agents[s].body_len > 0).collect();
    let d_free: Vec<f32> = (0..free.len().saturating_sub(1)).step_by(2).map(|i| (per(free[i]) - per(free[i + 1])).abs()).collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!("diferença média de energia por resíduo: entre dois LIGADOS {:.3} ({} pares) | entre dois soltos ao acaso {:.3} ({} pares)", mean(&d_link), d_link.len(), mean(&d_free), d_free.len());
    println!("(partilha de energia pela ligação: {} por passo; sinais: {})", w.params.bond_energy_share, w.params.bond_signal);
}
