//! Corre uma cena sem janela e mostra a população ao longo do tempo: para
//! ver se uma mudança (tabelas, regras) deixa viver a população de uma cena
//! antes de a aplicar ao mundo a correr. SCENE (por omissão o autosave),
//! STEPS (por omissão 6000), EVERY (por omissão 1000).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    // CLOCK_MUTE: silencia os relógios (0 = normais, 1 = mudos).
    w.params.clock_mute = envf("CLOCK_MUTE", 0.0);
    // ANGLE: multiplicador dos ângulos de repouso (por omissão o da cena).
    w.params.rest_angle_mult = envf("ANGLE", w.params.rest_angle_mult);
    // MODE: modo dos sinais (por omissão o da cena).
    w.params.signal_mode = envf("MODE", w.params.signal_mode);
    // PARAMS=nome=valor,nome=valor: muda parâmetros por nome (como o MCP).
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(w.params.set_named(k, v.parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
        w.invalidate_light();
    }
    let (steps, every) = (envf("STEPS", 6000.0) as u32, envf("EVERY", 1000.0) as u32);
    let report = |w: &World, done: u32| {
        let a = w.read_agents_blocking(&gpu);
        let alive: Vec<_> = a.iter().filter(|a| a.alive != 0).collect();
        let c = w.life_counters_blocking(&gpu);
        let e = alive.iter().map(|a| a.energy).sum::<f32>() / alive.len().max(1) as f32;
        println!("+{done:5}: vivos {:6}  energia média {e:6.1}  nascimentos {}  mortes {} (fome {})", alive.len(), c.births, c.deaths, c.starved);
    };
    report(&w, 0);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        if done % every < k || done == steps {
            report(&w, done);
        }
    }
    // No fim: % de agentes com cada órgão e o deslocamento em 200 passos,
    // por ter ou não relógio / sensor de comida direcional.
    let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let mut start = std::collections::HashMap::new();
    let mut count = [0u32; 32];
    let mut alive = 0u32;
    for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let mut has = [false; 32];
        for r in 0..a.body_len as usize {
            let o = (organs[slot * 32 + r / 2] >> ((r % 2) * 16)) & 0xFFFF;
            if o != 0 {
                has[((o & 0x1F) - 1) as usize] = true;
            }
        }
        alive += 1;
        for t in 0..32 {
            count[t] += has[t] as u32;
        }
        start.insert(a.id, (a.pos_x, a.pos_y, has[5], has[8], has[10]));
    }
    for (t, name) in ribossome::life::organs::ORGAN_NAMES.iter().enumerate() {
        println!("  {:5.1}% {name}", 100.0 * count[t] as f32 / alive.max(1) as f32);
    }
    let mut left = 200;
    while left > 0 {
        let k = MAX_STEPS_PER_FRAME.min(left);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        left -= k;
    }
    let wpc = WorldConfig::DEFAULT.world_units_per_cell as f32;
    let mut sums = std::collections::BTreeMap::new();
    for a in w.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0) {
        if let Some(&(x, y, clock, food, phys)) = start.get(&a.id) {
            let e = sums.entry((clock, food, phys)).or_insert((0f32, 0u32));
            e.0 += ((a.pos_x - x).powi(2) + (a.pos_y - y).powi(2)).sqrt() / wpc;
            e.1 += 1;
        }
    }
    println!("deslocamento em 200 passos (células), por (relógio, sensor comida dir., sensor físico dir.):");
    for ((c, f, ph), (d, n)) in sums {
        println!("  relógio {c:5} comida {f:5} físico {ph:5}: {n:5} agentes, {:.2} células", d / n as f32);
    }
}
