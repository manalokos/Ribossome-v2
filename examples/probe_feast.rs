//! Porque não vingam as proteases num mundo cheio de presas? Carrega uma cena
//! (qualquer tamanho de mundo) e mede as duas metades do problema:
//! - AS PRESAS: que fração dos agentes cada família de protease consegue
//!   morder (os que têm resíduos-alvo dela), que fração é imune às três
//!   especialistas, e quanto vale em média uma mordida de cada família
//!   (fração de alvos × defesa da prolina × inibidor), contra a generalista;
//! - OS PREDADORES: quantos há de cada família e de cada modo (contacto ou
//!   lítica), com que força (variante × intensidade), que fração das presas
//!   à sua volta conseguem morder de facto, e a energia que têm.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Scene, World};

/// Família que o bolso de um vizinho reconhece (pocket_family no shader).
fn pocket(letter: &str) -> usize {
    match letter {
        "D" | "E" => 1,
        "K" | "R" => 2,
        "F" | "L" | "W" | "Y" | "I" | "V" => 3,
        _ => 0,
    }
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
    let organs: Vec<u16> = bytemuck::pod_collect_to_vec(&gpu.read_buffer_blocking(&w.organs_buf));
    let bodies: Vec<u8> = gpu.read_buffer_blocking(&w.bodies_buf);
    let target: Vec<u32> = w.amino.iter().map(|r| r.protease_alvo as u32).collect();
    let letters: Vec<String> = w.amino.iter().map(|r| r.letra.clone()).collect();
    let pro = letters.iter().position(|l| l == "P").unwrap();
    let prot = &w.organ_table[11];
    let var = |v: usize, k: &str| prot.variantes.get(v).and_then(|m| m.get(k)).copied().unwrap_or(0.0);

    // ---- as presas ----
    let mut alive = 0u32;
    let (mut any, mut naked) = ([0u32; 3], 0u32);
    let mut immune3 = 0u32;
    let mut value = [0f64; 4]; // valor médio de uma mordida das famílias 1, 2, 3 e da generalista (por unidade de força)
    let mut e_sum = 0f64;
    // exposição de cada agente a cada família (para os predadores, abaixo)
    let mut exposed: Vec<[f32; 4]> = vec![[0.0; 4]; agents.len()];
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
        alive += 1;
        e_sum += a.energy as f64;
        let n = (a.body_len as usize).min(64);
        if n == 0 {
            naked += 1;
            exposed[slot] = [0.0, 0.0, 0.0, 1.0];
            value[3] += 1.0 / 3.0;
            immune3 += 1;
            continue;
        }
        let mut t = [0f32; 3];
        let mut prol = 0f32;
        let mut inhib = [0f32; 3];
        for k in 0..n {
            let aa = bodies[slot * 64 + k] as usize;
            for f in 0..3 {
                t[f] += ((target[aa] >> f) & 1) as f32;
            }
            prol += (aa == pro) as u32 as f32;
            let c = organs[slot * 64 + k];
            if c != 0 && (c & 0x1F) - 1 == 21 {
                // inibidor: a família é a do vizinho seguinte
                let fam = if k + 1 < n { pocket(&letters[bodies[slot * 64 + k + 1] as usize]) } else { 0 };
                if fam > 0 {
                    inhib[fam - 1] = 1.0;
                }
            }
        }
        let tot: f32 = inhib.iter().sum();
        if tot > 1.0 {
            for x in &mut inhib {
                *x /= tot;
            }
        }
        let resist = 1.0 - 0.9 * prol / n as f32;
        let mut hit = false;
        for f in 0..3 {
            let ex = t[f] / n as f32 * (1.0 - inhib[f]);
            exposed[slot][f] = ex * resist;
            if ex > 0.0 {
                any[f] += 1;
                hit = true;
            }
            value[f] += (ex * resist) as f64;
        }
        exposed[slot][3] = resist;
        value[3] += (resist / 3.0) as f64;
        immune3 += (!hit) as u32;
    }
    let pct = |k: u32| 100.0 * k as f64 / alive.max(1) as f64;
    println!("epoch {}: {alive} vivos, energia média {:.2}, {naked} sem corpo; protease_power {}, direta {}", w.params.epoch, e_sum / alive.max(1) as f64, w.params.protease_power, w.params.protease_direct);
    println!("AS PRESAS: que fração dos agentes cada protease consegue morder, e quanto vale a mordida (alvos × defesas, por unidade de força)");
    for (f, name) in ["família 1 (lisina, arginina)", "família 2 (aspartato, asparagina)", "família 3 (aromáticos, leucina)"].iter().enumerate() {
        println!("  {name:<34} morde {:>5.1}% dos agentes; valor médio {:.3}", pct(any[f]), value[f] / alive.max(1) as f64);
    }
    println!("  {:<34} morde 100.0% dos agentes; valor médio {:.3}", "generalista (um terço da força)", value[3] / alive.max(1) as f64);
    println!("  IMUNES às três especialistas: {:.1}% dos agentes", pct(immune3));

    // ---- os predadores ----
    println!("OS PREDADORES (por família e modo): agentes, força média (variante × intensidade), ligada por sinal, com boca, energia");
    let mut rows: std::collections::BTreeMap<(usize, bool), (u32, f64, u32, u32, f64)> = Default::default();
    for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0 && a.body_len > 0) {
        let n = (a.body_len as usize).min(64);
        let has_mouth = (0..n).any(|k| organs[slot * 64 + k] != 0 && (organs[slot * 64 + k] & 0x1F) - 1 == 0);
        for k in 0..n {
            let c = organs[slot * 64 + k];
            if c == 0 || (c & 0x1F) - 1 != 11 {
                continue;
            }
            let v = ((c >> 5) & 7) as usize;
            let gain = 2f32.powf(((c >> 8) as f32 - 32.0) / 8.0);
            let fam = if k + 1 < n { pocket(&letters[bodies[slot * 64 + k + 1] as usize]) } else { 0 };
            let e = rows.entry((fam, var(v, "alcance") >= 10.0)).or_default();
            e.0 += 1;
            e.1 += (var(v, "forca") * gain) as f64;
            e.2 += (var(v, "canal") >= 0.0) as u32;
            e.3 += has_mouth as u32;
            e.4 += a.energy as f64;
        }
    }
    for ((fam, lytic), (c, force, gated, mouth, en)) in &rows {
        let k = *c as f64;
        let fname = ["generalista", "família 1", "família 2", "família 3"][*fam];
        println!("  {fname:<12} {:<9}: {c:>6} proteases, força ×{:.2}, {:.0}% ligadas por sinal, {:.0}% com boca, energia {:.2}", if *lytic { "lítica" } else { "contacto" }, force / k, 100.0 * *gated as f64 / k, 100.0 * *mouth as f64 / k, en / k);
    }
    if rows.is_empty() {
        println!("  (não há nenhuma protease neste mundo)");
    }
}
