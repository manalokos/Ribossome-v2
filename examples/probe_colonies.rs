//! Colónias e migração. Carrega uma cena (por omissão o autosave) e:
//! 1. divide o mundo em blocos de 64 células e junta os blocos povoados
//!    (>= MIN agentes) que se tocam em COLÓNIAS; para cada uma mostra onde
//!    está, quantos agentes tem e as espécies principais (agrupamento dos
//!    genomas, ver `ribossome::species`);
//! 2. diz que espécies aparecem em mais de uma colónia (houve migração?);
//! 3. segue todos os agentes STEPS passos: quanto vivem, quanto se deslocam,
//!    e quantos acabam numa colónia diferente daquela onde começaram.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::species::cluster;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    // PARAMS=nome=valor,...: muda parâmetros antes de seguir os agentes.
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(w.params.set_named(k, v.parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
    }
    let wpc = cfg.world_units_per_cell as f32;
    const B: usize = 64; // células por bloco
    let nb = cfg.grid_size as usize / B;
    let min = envf("MIN", 25.0) as u32;
    let agents = w.read_agents_blocking(&gpu);
    let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
    let block_of = |x: f32, y: f32| -> usize {
        let bx = ((x / wpc) as usize / B).min(nb - 1);
        let by = ((y / wpc) as usize / B).min(nb - 1);
        by * nb + bx
    };
    let mut count = vec![0u32; nb * nb];
    for a in agents.iter().filter(|a| a.alive != 0) {
        count[block_of(a.pos_x, a.pos_y)] += 1;
    }
    // Componentes ligadas dos blocos povoados (8 vizinhos).
    let mut colony = vec![usize::MAX; nb * nb];
    let mut n_col = 0;
    for start in 0..nb * nb {
        if count[start] < min || colony[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        colony[start] = n_col;
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % nb) as i32, (i / nb) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= nb as i32 || yy >= nb as i32 {
                        continue;
                    }
                    let j = yy as usize * nb + xx as usize;
                    if count[j] >= min && colony[j] == usize::MAX {
                        colony[j] = n_col;
                        stack.push(j);
                    }
                }
            }
        }
        n_col += 1;
    }
    // Espécies (globais) e a espécie de cada agente.
    let alive: Vec<(usize, &ribossome::params::Agent)> = agents.iter().enumerate().filter(|(_, a)| a.alive != 0).collect();
    let genomes: Vec<Vec<u8>> = alive
        .iter()
        .map(|(slot, a)| (0..a.gene_len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect())
        .collect();
    let species = cluster(&genomes, 0.15);
    // Atribui cada genoma à espécie do líder mais próximo, pelo mesmo critério (reutiliza o agrupamento: mapa genoma -> índice).
    let mut sp_of: HashMap<&[u8], usize> = HashMap::new();
    for (si, s) in species.iter().enumerate() {
        sp_of.insert(s.leader.as_slice(), si);
    }
    // Para os não-líderes: a espécie cujo líder tem o comprimento mais próximo e mais bases iguais no início (aproximação barata).
    let assign = |g: &[u8]| -> usize {
        if let Some(&i) = sp_of.get(g) {
            return i;
        }
        let mut best = (0usize, i64::MIN);
        for (si, s) in species.iter().enumerate().take(40) {
            let same = g.iter().zip(&s.leader).filter(|(a, b)| a == b).count() as i64;
            let rc: Vec<u8> = g.iter().rev().map(|b| b ^ 1).collect();
            let same_rc = rc.iter().zip(&s.leader).filter(|(a, b)| a == b).count() as i64;
            let score = same.max(same_rc) - (g.len() as i64 - s.leader.len() as i64).abs();
            if score > best.1 {
                best = (si, score);
            }
        }
        best.0
    };
    struct Col {
        n: u32,
        x: f64,
        y: f64,
        sp: HashMap<usize, u32>,
        blocks: u32,
    }
    let mut cols: Vec<Col> = (0..n_col).map(|_| Col { n: 0, x: 0.0, y: 0.0, sp: HashMap::new(), blocks: 0 }).collect();
    for (i, c) in colony.iter().enumerate() {
        if *c != usize::MAX {
            cols[*c].blocks += 1;
            let _ = i;
        }
    }
    let mut loose = 0u32;
    let mut home: HashMap<u32, (usize, f32, f32)> = HashMap::new();
    for ((_, a), g) in alive.iter().zip(&genomes) {
        let c = colony[block_of(a.pos_x, a.pos_y)];
        home.insert(a.id, (c, a.pos_x, a.pos_y));
        if c == usize::MAX {
            loose += 1;
            continue;
        }
        let col = &mut cols[c];
        col.n += 1;
        col.x += a.pos_x as f64;
        col.y += a.pos_y as f64;
        *col.sp.entry(assign(g)).or_default() += 1;
    }
    let total = alive.len();
    println!("epoch {}, {total} agentes; {n_col} colónias (blocos de {B} células com >= {min} agentes); {loose} agentes fora de colónias ({:.1}%)", w.params.epoch, 100.0 * loose as f32 / total.max(1) as f32);
    let mut order: Vec<usize> = (0..n_col).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(cols[c].n));
    let mut where_sp: HashMap<usize, Vec<usize>> = HashMap::new();
    for (rank, &c) in order.iter().enumerate().take(12) {
        let col = &cols[c];
        let (cx, cy) = (col.x / col.n.max(1) as f64 / wpc as f64, col.y / col.n.max(1) as f64 / wpc as f64);
        let mut sp: Vec<(usize, u32)> = col.sp.iter().map(|(k, v)| (*k, *v)).collect();
        sp.sort_by_key(|x| std::cmp::Reverse(x.1));
        let top: Vec<String> = sp.iter().take(4).map(|(k, v)| format!("#{k} {:.0}%", 100.0 * *v as f32 / col.n as f32)).collect();
        for (k, v) in &sp {
            if *v as f32 >= 0.05 * col.n as f32 {
                where_sp.entry(*k).or_default().push(rank);
            }
        }
        println!("  colónia {rank}: {:5} agentes, {:3} blocos, centro ({:4.0}, {:4.0}) células [x, altura]; espécies: {}", col.n, col.blocks, cx, cy, top.join(", "));
    }
    let shared: Vec<String> = where_sp.iter().filter(|(_, v)| v.len() > 1).map(|(k, v)| format!("#{k} nas colónias {v:?}")).collect();
    println!("espécies com >= 5% em mais de uma colónia: {}", if shared.is_empty() { "nenhuma".to_string() } else { shared.join("; ") });
    // Distância entre os centros das maiores.
    let centers: Vec<(f64, f64)> = order.iter().take(6).map(|&c| (cols[c].x / cols[c].n.max(1) as f64 / wpc as f64, cols[c].y / cols[c].n.max(1) as f64 / wpc as f64)).collect();
    let mut dmin = f64::MAX;
    for i in 0..centers.len() {
        for j in i + 1..centers.len() {
            dmin = dmin.min(((centers[i].0 - centers[j].0).powi(2) + (centers[i].1 - centers[j].1).powi(2)).sqrt());
        }
    }
    println!("distância mínima entre centros das 6 maiores: {dmin:.0} células");

    // 3. Seguir os agentes.
    let steps = envf("STEPS", 2000.0) as u32;
    let c0 = w.life_counters_blocking(&gpu);
    let mut last: HashMap<u32, (f32, f32, u32)> = home.iter().map(|(id, v)| (*id, (v.1, v.2, 0))).collect();
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done).min(100);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        for a in w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0) {
            if let Some(l) = last.get_mut(&a.id) {
                *l = (a.pos_x, a.pos_y, done);
            }
        }
    }
    let c1 = w.life_counters_blocking(&gpu);
    let deaths = c1.deaths.wrapping_sub(c0.deaths) as f32;
    let (mut moved, mut far, mut switched, mut survived) = (Vec::new(), 0u32, 0u32, 0u32);
    // Quem mudou de colónia: (origem, destino) -> quantos, pela ordem de tamanho.
    let rank_of: HashMap<usize, usize> = order.iter().enumerate().map(|(r, &c)| (c, r)).collect();
    let mut trips: HashMap<(usize, usize), u32> = HashMap::new();
    let (mut loose_alive, mut loose_total, mut loose_arrived) = (0u32, 0u32, 0u32);
    for (id, (c, x, y)) in &home {
        let (lx, ly, seen) = last[id];
        let d = ((lx - x).powi(2) + (ly - y).powi(2)).sqrt() / wpc;
        moved.push(d);
        far += (d > 64.0) as u32;
        survived += (seen >= steps) as u32;
        let c_end = colony[block_of(lx, ly)];
        if *c != usize::MAX && c_end != usize::MAX && c_end != *c {
            switched += 1;
            *trips.entry((rank_of[c], rank_of[&c_end])).or_default() += 1;
        }
        if *c == usize::MAX {
            loose_total += 1;
            loose_alive += (seen >= steps) as u32;
            loose_arrived += (c_end != usize::MAX) as u32;
        }
    }
    moved.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f32| moved[((moved.len() - 1) as f32 * f) as usize];
    println!(
        "em {steps} passos: sobrevivem {:.1}%; vida média ~{:.0} passos; deslocamento até morrer ou ao fim: mediana {:.1}, 90% {:.1}, 99% {:.1}, máximo {:.1} células",
        100.0 * survived as f32 / total.max(1) as f32,
        total as f32 * steps as f32 / deaths.max(1.0),
        q(0.5),
        q(0.9),
        q(0.99),
        q(1.0)
    );
    println!("  andaram mais de 64 células: {far}; acabaram num bloco de outra colónia: {switched}");
    let mut t: Vec<_> = trips.into_iter().collect();
    t.sort_by_key(|x| std::cmp::Reverse(x.1));
    println!("  viagens (colónia de origem -> destino: agentes): {}", t.iter().take(10).map(|((a, b), n)| format!("{a}->{b}: {n}")).collect::<Vec<_>>().join(", "));
    println!("  dos {loose_total} que começaram fora de colónias: {loose_alive} vivos ao fim, {loose_arrived} acabaram dentro de uma colónia");
    // Energia: dentro e fora das colónias, no início.
    let (mut e_in, mut n_in, mut e_out, mut n_out) = (0f32, 0u32, 0f32, 0u32);
    for (_, a) in &alive {
        if colony[block_of(a.pos_x, a.pos_y)] == usize::MAX {
            e_out += a.energy;
            n_out += 1;
        } else {
            e_in += a.energy;
            n_in += 1;
        }
    }
    println!("  energia média no início: {:.1} dentro de colónias, {:.1} fora", e_in / n_in.max(1) as f32, e_out / n_out.max(1) as f32);
}
