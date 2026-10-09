//! DERIVA DE GRUPOS LIGADOS: há grupos de agentes agarrados por âncoras que
//! andam sem nadar? Carrega uma cena, desliga tudo o que move um corpo de
//! fora ou por dentro (corrente, agitação térmica, natação, nascimentos e
//! mortes) e mede quanto anda o centro de cada grupo ligado, por tamanho do
//! grupo. Com as forças entre agentes recíprocas, nenhum grupo devia andar.
//! SCENE (por omissão o autosave), STEPS (2000), PARAMS=nome=valor,...
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{BOND_STRIDE, MAX_STEPS_PER_FRAME, Scene, World};
use std::collections::HashMap;

fn find(p: &mut Vec<usize>, x: usize) -> usize {
    let mut r = x;
    while p[r] != r {
        r = p[r];
    }
    let mut c = x;
    while p[c] != r {
        let n = p[c];
        p[c] = r;
        c = n;
    }
    r
}

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    for (k, v) in [
        ("flow_coupling", 0.0),
        ("thermal_kt", 0.0),
        ("brownian", 0.0),
        ("brownian_rot", 0.0),
        ("swim_gain", 0.0),
        ("swim_wobble", 0.0),
        ("motor_amplitude", 0.0),
        ("clock_mute", 1.0),
        ("death_probability", 0.0),
        ("pairing_rate", 0.0),
        ("uv_damage", 0.0),
        ("heat_kill", 0.0),
        ("maintenance_cost", 0.0),
        ("bond_rate", 0.0),
    ] {
        assert!(w.params.set_named(k, v), "parâmetro desconhecido: {k}");
    }
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(w.params.set_named(k, v.parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
    }
    let run = |w: &mut World, n: u32| {
        let mut done = 0;
        while done < n {
            let k = MAX_STEPS_PER_FRAME.min(n - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += k;
        }
    };
    // Uns passos para as poses assentarem com os sinais já parados.
    run(&mut w, 256);
    let before = w.read_agents_blocking(&gpu);
    let bonds: Vec<[u32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.bonds_buf)).to_vec();
    let stride = BOND_STRIDE as usize;
    let mut parent: Vec<usize> = (0..before.len()).collect();
    for slot in 0..before.len() {
        if before[slot].alive == 0 {
            continue;
        }
        for b in &bonds[slot * stride..slot * stride + stride - 1] {
            if b[0] != u32::MAX && (b[0] as usize) < before.len() && before[b[0] as usize].alive != 0 {
                let (x, y) = (find(&mut parent, slot), find(&mut parent, b[0] as usize));
                parent[x] = y;
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for slot in 0..before.len() {
        if before[slot].alive != 0 {
            let r = find(&mut parent, slot);
            groups.entry(r).or_default().push(slot);
        }
    }
    run(&mut w, steps);
    let after = w.read_agents_blocking(&gpu);
    // tamanho do grupo -> deslocamentos do centro (unidades do mundo).
    let mut by_size: std::collections::BTreeMap<usize, Vec<(f32, usize)>> = Default::default();
    for (root, g) in &groups {
        if g.iter().any(|&s| after[s].alive == 0 || after[s].id != before[s].id) {
            continue;
        }
        let c = |a: &[ribossome::params::Agent]| {
            let n = g.len() as f32;
            (g.iter().map(|&s| a[s].pos_x).sum::<f32>() / n, g.iter().map(|&s| a[s].pos_y).sum::<f32>() / n)
        };
        let (b, a) = (c(&before), c(&after));
        by_size.entry(g.len().min(6)).or_default().push((((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt(), *root));
    }
    println!("{steps} passos sem corrente, sem agitação e sem natação: deslocamento do centro de cada grupo (unidades do mundo)");
    println!("{:>8} {:>8} {:>10} {:>10} {:>10} {:>12}", "grupo", "quantos", "mediana", "p90", "máximo", "> 20 un. (%)");
    for (size, v) in by_size.iter_mut() {
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let q = |f: f32| v[((v.len() - 1) as f32 * f) as usize].0;
        let far = v.iter().filter(|d| d.0 > 20.0).count() as f32 / v.len() as f32 * 100.0;
        println!("{:>8} {:>8} {:>10.2} {:>10.2} {:>10.2} {:>12.1}", if *size == 6 { "6+".to_string() } else { size.to_string() }, v.len(), q(0.5), q(0.9), q(1.0), far);
    }
    // Os grupos que mais andaram, com quem os compõe.
    let mut all: Vec<(f32, usize)> = by_size.iter().filter(|(s, _)| **s > 1).flat_map(|(_, v)| v.iter().copied()).collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (d, root) in all.iter().take(6) {
        let g = &groups[root];
        println!("  andou {d:.1}: {} agentes, raios {:?}, resíduos {:?}", g.len(), g.iter().map(|&s| before[s].radius.round()).collect::<Vec<_>>(), g.iter().map(|&s| before[s].body_len).collect::<Vec<_>>());
    }
}
