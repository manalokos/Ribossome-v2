//! Diagnóstico: estatísticas do fluido por linha (bordas vs interior).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = if std::env::var("BIG").is_ok() { WorldConfig::DEFAULT } else { WorldConfig::TEST };
    let mut world = World::new(&gpu, cfg, 3);
    world.settings.jacobi_iters = std::env::var("J").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    world.seed_matter(&gpu, 3);
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(640);
    let t0 = std::time::Instant::now();
    for _ in 0..(steps / MAX_STEPS_PER_FRAME) {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    }
    let el = t0.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!("{:.3} ms por passo (J={})", el, world.settings.jacobi_iters);
    let vel = world.read_f32_blocking(&gpu, &world.velocity_buf);
    let n = cfg.fluid_size as usize;
    let vy = |x: usize, y: usize| vel[(y * n + x) * 2 + 1];
    let vx = |x: usize, y: usize| vel[(y * n + x) * 2];
    for &y in &[n / 4, n / 2, 3 * n / 4] {
        let m: f32 = (0..n).map(|x| vy(x, y)).sum::<f32>() / n as f32;
        let a: f32 = (0..n).map(|x| vy(x, y).abs()).sum::<f32>() / n as f32;
        println!("linha y={y:3}: vy média {m:8.3}  |vy| média {a:8.3}");
    }
    for &x in &[0, 1, n - 2, n - 1] {
        let a: f32 = (0..n).map(|y| vx(x, y).abs()).sum::<f32>() / n as f32;
        println!("coluna x={x:3}: |vx| média {a:8.3}");
    }
    let interior: f32 = (2..n - 2).flat_map(|y| (2..n - 2).map(move |x| (x, y))).map(|(x, y)| vy(x, y)).sum::<f32>()
        / ((n - 4) * (n - 4)) as f32;
    println!("interior vy média {interior:.4}");
}
