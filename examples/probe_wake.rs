//! Caudas sem monómeros junto ao terreno: conta as células de ÁGUA perto do
//! fundo (até 400 células acima do leito) quase vazias, ao longo do tempo.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::World;

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    if let Ok(t) = std::env::var("TERRAIN") {
        let hot = world.load_terrain_png(std::path::Path::new(&t)).unwrap();
        println!("terreno {t}: {hot} células quentes");
    }
    world.seed_matter(&gpu, 1);
    if let Ok(v) = std::env::var("PRESS") { world.params.monomer_pressure = v.parse().unwrap(); }
    let n = cfg.grid_size as usize;
    for round in 1..=6 {
        for _ in 0..16 {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            world.encode_steps(&gpu.queue, &mut enc, 64);
            gpu.queue.submit([enc.finish()]);
        }
        gpu.wait_idle();
        let cells = world.read_cells_blocking(&gpu);
        let g = world.read_gamma_blocking(&gpu);
        let (mut water, mut empty, mut low, mut sum) = (0u64, 0u64, 0u64, 0u64);
        let (mut rub, mut rub_sum) = (0u64, 0u64);
        for y in 0..400 {
            for x in 0..n {
                let i = y * n + x;
                let t: u32 = (0..4).map(|c| (cells[i * 4 + c] & 0xFFFF) + (cells[i * 4 + c] >> 16)).sum();
                if g[i] == 0 {
                    water += 1;
                    sum += t as u64;
                    if t == 0 { empty += 1; }
                    if t < 3 { low += 1; }
                } else if g[i] < 3 {
                    rub += 1;
                    rub_sum += t as u64;
                }
            }
        }
        // Água junto ao terreno (a <= 3 células de qualquer gamma) vs longe (> 12).
        let mut dist = vec![u32::MAX; n * n];
        let mut q = std::collections::VecDeque::new();
        for i in 0..n * n { if g[i] > 0 { dist[i] = 0; q.push_back(i); } }
        while let Some(i) = q.pop_front() {
            let d = dist[i];
            if d >= 13 { continue; }
            let (x, y) = (i % n, i / n);
            for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                if nx < n && ny < n && dist[ny * n + nx] > d + 1 { dist[ny * n + nx] = d + 1; q.push_back(ny * n + nx); }
            }
        }
        let (mut near_s, mut near_n, mut far_s, mut far_n, mut near_empty) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for y in 0..600 {
            for x in 0..n {
                let i = y * n + x;
                if g[i] != 0 { continue; }
                let t: u64 = (0..4).map(|c| ((cells[i * 4 + c] & 0xFFFF) + (cells[i * 4 + c] >> 16)) as u64).sum();
                if dist[i] <= 3 { near_s += t; near_n += 1; if t == 0 { near_empty += 1; } }
                else if dist[i] > 12 { far_s += t; far_n += 1; }
            }
        }
        // Uniformidade: desvio-padrão relativo da densidade da água (mundo todo).
        let (mut s1, mut s2, mut cnt) = (0f64, 0f64, 0f64);
        for i in 0..n * n {
            if g[i] != 0 { continue; }
            let t: f64 = (0..4).map(|c| ((cells[i * 4 + c] & 0xFFFF) + (cells[i * 4 + c] >> 16)) as f64).sum();
            s1 += t; s2 += t * t; cnt += 1.0;
        }
        let mean = s1 / cnt;
        println!("  água: média {mean:.2}, desvio relativo {:.3}", ((s2 / cnt - mean * mean).max(0.0)).sqrt() / mean);
        println!("  junto ao terreno: média {:.2} ({:.2}% vazias); longe: média {:.2}", near_s as f64 / near_n as f64, near_empty as f64 / near_n as f64 * 100.0, far_s as f64 / far_n as f64);
        println!(
            "passo {:>5}: água perto do fundo {water}: vazias {empty} ({:.2}%), <3 {low} ({:.2}%), média {:.2}; entulho {rub} células, média {:.2}",
            round * 1024, empty as f64 / water as f64 * 100.0, low as f64 / water as f64 * 100.0, sum as f64 / water as f64, rub_sum as f64 / rub.max(1) as f64
        );
    }
}
