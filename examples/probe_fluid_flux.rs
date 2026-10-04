//! O campo de velocidades conserva volume? Num fluido incompressível num
//! aquário fechado, o caudal vertical que atravessa cada linha horizontal é
//! ZERO (o que sobe tem de descer). Carrega uma cena, corre STEPS passos e
//! mede, por faixas de altura, a velocidade vertical média (em células do
//! ambiente por passo) e a divergência residual média |div| (diferenças
//! centrais, como o próprio fluido a calcula). SUBSTEP e MG como em
//! probe_matter_height.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    if let Some(c) = env("MG") {
        w.settings.mg_cycles = c as u32;
    }
    if let Some(c) = env("SUBSTEP") {
        w.settings.fluid_substep = c as u32;
    }
    let steps = env("STEPS").unwrap_or(2000.0) as u32;
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let f = cfg.fluid_size as usize;
    let vel: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.velocity_buf)).to_vec();
    // Velocidade (células do fluido/s) -> células do ambiente por passo.
    let scale = cfg.grid_size as f32 / cfg.fluid_size as f32 * w.params.dt;
    let bands = 8;
    println!("ciclos V {}, resolve de {} em {} passos; faixas de cima para baixo:", w.settings.mg_cycles, w.settings.fluid_substep, w.settings.fluid_substep);
    let (mut all_vy, mut all_up) = (0f64, 0f64);
    for b in (0..bands).rev() {
        let (mut vy, mut up, mut div, mut n) = (0f64, 0f64, 0f64, 0u32);
        for y in b * f / bands..(b + 1) * f / bands {
            for x in 0..f {
                let v = vel[y * f + x];
                vy += v[1] as f64;
                up += v[1].abs() as f64;
                if x > 0 && x + 1 < f && y > 0 && y + 1 < f {
                    div += (0.5 * ((vel[y * f + x + 1][0] - vel[y * f + x - 1][0]) + (vel[(y + 1) * f + x][1] - vel[(y - 1) * f + x][1]))).abs() as f64;
                }
                n += 1;
            }
        }
        all_vy += vy;
        all_up += up;
        println!(
            "  velocidade vertical média {:+.5} células/passo (|v_y| médio {:.5}); |div| médio {:.5} por passo",
            vy / n as f64 * scale as f64,
            up / n as f64 * scale as f64,
            div / n as f64 * w.params.dt as f64
        );
    }
    println!("mundo: v_y médio {:+.5} células/passo; desequilíbrio = {:.1}% do movimento vertical", all_vy / (f * f) as f64 * scale as f64, 100.0 * all_vy / all_up.max(1e-12));
}
