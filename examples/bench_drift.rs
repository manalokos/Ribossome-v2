//! O passo do mundo completo fica mais lento com o tempo? Mede ms/passo por
//! janelas ao longo de STEPS passos. SEEDS=0 sem agentes; FLUID=0, TERRAIN=0
//! desligam sistemas.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    world.seed_matter(&gpu, 1);
    world.params.death_probability = 0.14;
    world.params.uv_damage = 10.0;
    world.settings.fluid_enabled = env("FLUID", 1) != 0;
    world.settings.terrain_enabled = env("TERRAIN", 1) != 0;
    let seeds: u32 = env("SEEDS", 500);
    if seeds > 0 {
        let mut rng = ribossome::life::SplitMix(3);
        world.request_seeds(&ribossome::life::seed_requests(seeds, [30, 200], true, cfg.sim_size(), &mut rng));
    }
    let total: u32 = env("STEPS", 10_000);
    let window: u32 = env("WINDOW", 1000);
    let mut done = 0;
    while done < total {
        let t = std::time::Instant::now();
        let mut w = 0;
        while w < window {
            let k = MAX_STEPS_PER_FRAME.min(window - w);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            world.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            w += k;
        }
        gpu.wait_idle();
        done += window;
        let ms = t.elapsed().as_secs_f64() * 1000.0 / window as f64;
        let alive = world.life_counters_blocking(&gpu).alive(cfg.max_agents);
        let cells = world.read_cells_blocking(&gpu);
        let n = cfg.grid_size as usize;
        let mut occupied = 0usize;
        let mut maxc = 0u32;
        let mut over24 = 0usize;
        let mut border = 0u64;
        let mut argmax = 0usize;
        let mut total = 0u64;
        for i in 0..n * n {
            let c: u32 = (0..4).map(|ch| (cells[i * 4 + ch] & 0xFFFF) + (cells[i * 4 + ch] >> 16)).sum();
            if c > 0 { occupied += 1; }
            if c > maxc { maxc = c; argmax = i; }
            if c > 24 { over24 += 1; }
            let (x, y) = (i % n, i / n);
            if x < 2 || y < 2 || x >= n - 2 || y >= n - 2 { border += c as u64; }
            total += c as u64;
        }
        let gamma: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&world.gamma_buf)).to_vec();
        let (ax, ay) = (argmax % n, argmax / n);
        let around: Vec<u32> = [(0i64, 1i64), (0, -1), (1, 0), (-1, 0)].iter().map(|(dx, dy)| {
            let (x, y) = ((ax as i64 + dx).clamp(0, n as i64 - 1) as usize, (ay as i64 + dy).clamp(0, n as i64 - 1) as usize);
            gamma[y * n + x]
        }).collect();
        println!("  célula máx em ({ax}, {ay}) canais {:?}  terreno aqui {}  vizinhos (cima, baixo, dir, esq) {:?}",
            (0..4).map(|ch| (cells[argmax * 4 + ch] & 0xFFFF, cells[argmax * 4 + ch] >> 16)).collect::<Vec<_>>(), gamma[argmax], around);
        let vel = world.read_f32_blocking(&gpu, &world.velocity_buf);
        let mut vmax = 0.0f32;
        let mut vsum = 0.0f64;
        for v in vel.chunks(2) {
            let s = v[0].hypot(v[1]);
            vmax = vmax.max(s);
            vsum += s as f64;
        }
        println!(
            "passo {done:>6}: {ms:6.3} ms/passo  ({alive} agentes)  células ocupadas {occupied}  máx/célula {maxc}  células>24 {over24}  na borda {:.1}%  |v| média {:.3} máx {vmax:.2}",
            border as f64 / total as f64 * 100.0,
            vsum / (vel.len() / 2) as f64
        );
    }
}
