//! Velocidade média da água no entulho vs água livre (células do fluido),
//! com o terreno TERRAIN (por omissão o gerado), ao fim de STEPS passos.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::World;

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    if let Ok(t) = std::env::var("TERRAIN") {
        world.load_terrain_png(std::path::Path::new(&t)).unwrap();
    }
    world.seed_matter(&gpu, 1);
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let mut done = 0;
    while done < steps {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, 64);
        gpu.queue.submit([enc.finish()]);
        done += 64;
    }
    gpu.wait_idle();
    let vel = world.read_f32_blocking(&gpu, &world.velocity_buf);
    let g = world.read_gamma_blocking(&gpu);
    let (n, f) = (cfg.grid_size as usize, cfg.fluid_size as usize);
    let k = n / f;
    let mut acc = [(0.0f64, 0u64); 3]; // água, entulho 1, entulho 2
    for y in 0..f {
        for x in 0..f {
            let gg = g[(y * k) * n + x * k];
            let s = (vel[(y * f + x) * 2] as f64).hypot(vel[(y * f + x) * 2 + 1] as f64);
            let i = match gg { 0 => 0, 1 => 1, 2 => 2, _ => continue };
            acc[i].0 += s;
            acc[i].1 += 1;
        }
    }
    let m = |i: usize| acc[i].0 / acc[i].1.max(1) as f64;
    println!("|v| médio: água {:.3}, entulho 1 grão {:.3} ({:.0}%), 2 grãos {:.3} ({:.0}%)", m(0), m(1), m(1) / m(0) * 100.0, m(2), m(2) / m(0) * 100.0);
}
