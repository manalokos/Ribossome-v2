#![allow(clippy::needless_range_loop)]
//! Diagnóstico: deriva vertical dos monómeros e riscas horizontais.
//! Variáveis: STEPS, FLUID=0, DIFF (slider de difusão), FLAT=1 (sem terreno), BIG=1.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn report(tag: &str, cells: &[u32], n: usize) {
    let mut rows = vec![0f64; n];
    for y in 0..n {
        for x in 0..n {
            let i = (y * n + x) * 4;
            rows[y] += (0..4).map(|c| ((cells[i + c] & 0xFFFF) + (cells[i + c] >> 16)) as f64).sum::<f64>();
        }
    }
    let total: f64 = rows.iter().sum();
    let com: f64 = rows.iter().enumerate().map(|(y, &r)| y as f64 * r).sum::<f64>() / total;
    // Massa média por y mod 16, relativa à média (só nas linhas de água do meio).
    let mut m16 = [0f64; 16];
    let mut c16 = [0f64; 16];
    for y in n / 4..3 * n / 4 {
        m16[y % 16] += rows[y];
        c16[y % 16] += 1.0;
    }
    let avg: f64 = m16.iter().zip(&c16).map(|(m, c)| m / c).sum::<f64>() / 16.0;
    let prof: Vec<String> = m16.iter().zip(&c16).map(|(m, c)| format!("{:+.1}", (m / c / avg - 1.0) * 100.0)).collect();
    println!("{tag}: centro de massa y = {:.1} ({:.3} da altura)", com, com / n as f64);
    println!("    massa por y%16 (% vs média): {}", prof.join(" "));
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = if env("BIG", 0) == 1 { WorldConfig::DEFAULT } else { WorldConfig::TEST };
    let n = cfg.grid_size as usize;
    let mut world = World::new(&gpu, cfg, 5);
    world.settings.fluid_enabled = env("FLUID", 1) == 1;
    world.params.diffusion = env("DIFF", 20.0);
    world.params.settle = 0.0;
    world.seed_matter(&gpu, 5);
    if env("FLAT", 0) == 1 {
        gpu.queue.write_buffer(&world.gamma_buf, 0, &vec![0u8; (cfg.cells() * 4) as usize]);
    }
    report("início", &world.read_cells_blocking(&gpu), n);
    let steps: u32 = env("STEPS", 4096);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        if done % 1024 == 0 || done == steps {
            report(&format!("passo {done}"), &world.read_cells_blocking(&gpu), n);
        }
    }
}
