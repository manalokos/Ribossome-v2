//! A luz por propagação (1 linha por passo) converge para a da varredura
//! exata? Corre STEPS passos (sem agentes), lê a luz, força uma varredura e
//! compara.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::World;

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    world.seed_matter(&gpu, 1);
    // Terreno parado: a comparação é só da luz.
    world.settings.terrain_enabled = false;
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let mut done = 0;
    while done < steps {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, 64);
        gpu.queue.submit([enc.finish()]);
        done += 64;
    }
    gpu.wait_idle();
    let prop = world.read_f32_blocking(&gpu, &world.light_buf);
    world.invalidate_light();
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_steps(&gpu.queue, &mut enc, 1);
    gpu.queue.submit([enc.finish()]);
    let sweep = world.read_f32_blocking(&gpu, &world.light_buf);
    let ls = (cfg.grid_size / ribossome::shaders::LIGHT_DIV) as usize;
    let mut worst = 0.0f32;
    for i in 0..ls * ls {
        let rel = (prop[i] - sweep[i]).abs() / sweep[i].max(1e-6);
        if sweep[i] > 1e-5 {
            worst = worst.max(rel);
        }
    }
    println!("depois de {done} passos: diferença relativa máxima propagação vs varredura = {worst:.2e}");
}
