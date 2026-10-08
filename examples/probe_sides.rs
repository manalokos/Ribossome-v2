//! SENSORES DE UM LADO. Um agente com DOIS sensores direcionais de corpos
//! (Y + D), um com índice de intensidade par (lado esquerdo) e outro ímpar
//! (lado direito), e um molho de presas só de um dos lados do mundo (a
//! leste). Como cada agente nasce rodado ao acaso, em cada um só deve
//! disparar o sensor que ficou virado para as presas: conta-se, em 64
//! agentes, quantos têm só o par, só o ímpar, os dois ou nenhum a sentir.
use ribossome::gpu::Gpu;
use ribossome::life::organs::translate_organs;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

const B: [&str; 4] = ["A", "U", "G", "C"];

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 3);
    let code = ribossome::life::table::code_to_gpu(&w.organ_code);
    // Dois codões de intensidade: um de índice par e um de índice ímpar.
    let (mut even, mut odd) = (None, None);
    for i in 0..64usize {
        let cod = format!("{}{}{}", B[i / 16], B[(i / 4) % 4], B[i % 4]);
        let body = translate_organs(&bases(&format!("AUG UAU GAU {cod} GGU UAA")), true, &code);
        if body.len() != 3 {
            continue;
        }
        if let Some((8, _, g)) = body.get(1).and_then(|r| r.organ) {
            if g % 2 == 0 && (28..=36).contains(&g) && even.is_none() {
                even = Some((cod.clone(), g));
            }
            if g % 2 == 1 && (28..=36).contains(&g) && odd.is_none() {
                odd = Some((cod, g));
            }
        }
    }
    let ((ce, ge), (co, go)) = (even.expect("sem codão par"), odd.expect("sem codão ímpar"));
    println!("intensidade par: {ce} (índice {ge}); ímpar: {co} (índice {go})");
    // Dois tipos de agente, cada um com UM sensor (resíduo 1): par ou ímpar.
    let genomes = [bases(&format!("AUG UAU GAU {ce} {} UAA", "GGU ".repeat(8))), bases(&format!("AUG UAU GAU {co} {} UAA", "GGU ".repeat(8)))];
    let body = translate_organs(&genomes[0], true, &code);
    println!("corpo de {} resíduos, sensor no resíduo 1", body.len());
    let prey = bases(&format!("AUG {} UAA", "AAA ".repeat(12)));
    w.custom_terrain = Some((vec![0; cfg.cells() as usize], vec![0.0; cfg.cells() as usize]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.brownian_rot = 0.0;
    w.params.uv_strength = 0.0;
    w.params.sedimentation = 0.0;
    w.params.signal_crosstalk = 0.0;
    let n = 256;
    let mut reqs = Vec::new();
    for i in 0..n {
        let (x, y) = (4000.0 + 1500.0 * (i % 16) as f32, 4000.0 + 1500.0 * (i / 16) as f32);
        reqs.push(SpawnRequest::with_genome(x, y, &genomes[i % 2]));
        // Presas a leste, a 110 unidades (dentro do raio de 180 do sensor).
        for j in 0..3 {
            reqs.push(SpawnRequest::with_genome(x + 110.0, y - 40.0 + 40.0 * j as f32, &prey));
        }
    }
    w.request_seeds(&reqs);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    w.encode_steps(&gpu.queue, &mut enc, 4);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let agents = w.read_agents_blocking(&gpu);
    let sig: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
    let body_pos: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.body_pos_buf)).to_vec();
    // [par, ímpar] × [presas à esquerda, presas à direita] -> (quantos, quantos sentem)
    let mut tally = [[(0u32, 0.0f32); 2]; 2];
    for (slot, a) in agents.iter().enumerate() {
        if a.alive == 0 || a.body_len as usize != body.len() {
            continue;
        }
        let kind = ((((a.pos_x - 4000.0) / 1500.0).round() as i64 + 16 * ((a.pos_y - 4000.0) / 1500.0).round() as i64).rem_euclid(2)) as usize;
        let fires = (sig[slot * 64 + 1][0] + sig[slot * 64 + 1][1]).abs();
        // Lado onde ficaram as presas (leste) em relação à cadeia no sensor:
        // normal esquerda = (−ty, tx) da tangente rodada para o mundo.
        let (p0, p1) = (body_pos[slot * 64], body_pos[slot * 64 + 2]);
        let t = [p1[0] - p0[0], p1[1] - p0[1]];
        let (sn, cs) = a.rot.sin_cos();
        let tw = [cs * t[0] - sn * t[1], sn * t[0] + cs * t[1]];
        // Só os casos claros: a cadeia quase norte-sul (as presas bem de um lado).
        let len = (tw[0] * tw[0] + tw[1] * tw[1]).sqrt().max(1e-6);
        if (tw[1] / len).abs() < 0.8 {
            continue;
        }
        let side = if -tw[1] > 0.0 { 0 } else { 1 };
        tally[kind][side].0 += 1;
        tally[kind][side].1 += fires;
    }
    println!("{:28} {:>22} {:>22}", "sensor", "presas à ESQUERDA", "presas à DIREITA");
    for (kind, name) in ["índice par (lado esquerdo)", "índice ímpar (lado direito)"].iter().enumerate() {
        println!("{name:28} {:>14.3} (n={:<3}) {:>14.3} (n={:<3})", tally[kind][0].1 / tally[kind][0].0.max(1) as f32, tally[kind][0].0, tally[kind][1].1 / tally[kind][1].0.max(1) as f32, tally[kind][1].0);
    }
    println!("(cada célula: |sinal| médio no sensor; só agentes com a cadeia quase perpendicular às presas)");
}
