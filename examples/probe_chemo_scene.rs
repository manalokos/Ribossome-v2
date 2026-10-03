//! Quimiossintéticos numa cena real: carrega o autosave (ou SCENE), larga N
//! agentes desenhados (3 órgãos de quimiossíntese produtores, sem boca) ao
//! acaso na faixa de altura Y0..Y1 (frações do mundo) e conta, a cada 500
//! passos, quantos agentes com quimiossíntese há (os largados e os filhos),
//! a energia média deles e o redutor onde estão. Com emparelhamento ligado.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    w.load_scene(&gpu, &scene).unwrap();
    let g = cfg.grid_size as usize;
    let f = cfg.fluid_size as usize;
    let wpc = cfg.world_units_per_cell as f32;

    // N (AAU) + L (CUU) = quimiossíntese variante 0 (produtor).
    let text = format!("AUG {} {} UAA", "AAU CUU GGU ".repeat(3), "GGU ".repeat(6));
    let genome: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    let n = envf("N", 200.0) as u32;
    let (y0, y1) = (envf("Y0", 0.25), envf("Y1", 0.6));
    let mut rng = ribossome::life::SplitMix(11);
    let s = cfg.sim_size();
    let reqs: Vec<SpawnRequest> = (0..n)
        .map(|_| {
            let x = (0.02 + 0.96 * rng.f32()) * s;
            let y = (y0 + (y1 - y0) * rng.f32()) * s;
            SpawnRequest::with_genome(x, y, &genome)
        })
        .collect();

    let report = |w: &World, label: &str| {
        let agents = w.read_agents_blocking(&gpu);
        let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
        let redox = w.read_f32_blocking(&gpu, &w.redox_buf);
        let c = w.life_counters_blocking(&gpu);
        let (mut nc, mut e, mut r, mut alive) = (0u32, 0.0f32, 0.0f32, 0u32);
        for (slot, a) in agents.iter().enumerate().filter(|(_, a)| a.alive != 0) {
            alive += 1;
            let chemo = (0..a.body_len as usize).any(|k| {
                let o = (organs[slot * 32 + k / 2] >> ((k % 2) * 16)) & 0xFFFF;
                o != 0 && (o & 0xF) - 1 == 14
            });
            if chemo {
                nc += 1;
                e += a.energy;
                let (cx, cy) = (((a.pos_x / wpc) as usize).min(g - 1), ((a.pos_y / wpc) as usize).min(g - 1));
                r += redox[(cy * f / g) * f + cx * f / g];
            }
        }
        let d = nc.max(1) as f32;
        println!(
            "{label} (epoch {}): vivos {alive}, com quimiossíntese {nc}, energia média {:.1}, redutor onde estão {:.1}; mortes {} (fome {})",
            w.params.epoch,
            e / d,
            r / d,
            c.deaths,
            c.starved
        );
    };
    report(&w, "antes");
    w.request_seeds(&reqs);
    let steps = envf("STEPS", 3000.0) as u32;
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        if done % 512 == 0 || done == steps || done == 64 {
            report(&w, &format!("+{done}"));
        }
    }
}
