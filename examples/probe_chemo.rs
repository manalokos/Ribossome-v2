//! Quimiossíntese: agentes desenhados (3 órgãos de quimiossíntese
//! produtores, sem boca) a várias distâncias da fumarola mais quente do
//! terreno do projeto. Mostra temperatura e redutor a cada distância e quantos
//! sobrevivem 1500 passos (com a desnaturação pelo calor ligada).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 2);
    w.load_terrain_png(std::path::Path::new("assets/terreno.png")).unwrap();
    w.seed_matter(&gpu, 2);
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    if let Some(v) = envf("DENAT") {
        w.params.denature_temp = v;
    }
    if let Some(v) = envf("HEAT") {
        w.params.heat_kill = v;
    }
    if let Some(v) = envf("DECAY") {
        w.params.redox_decay = v;
    }
    let run = |w: &mut World, steps: u32| {
        let mut done = 0;
        while done < steps {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += MAX_STEPS_PER_FRAME;
        }
    };
    // Deixa o redutor acumular.
    run(&mut w, 3000);
    let heat = w.heat_grid();
    let g = cfg.grid_size as usize;
    let (hot, _) = heat.iter().enumerate().fold((0, 0.0f32), |b, (i, &h)| if h > b.1 { (i, h) } else { b });
    let (hx, hy) = ((hot % g) as f32 + 0.5, (hot / g) as f32 + 0.5);
    let wpc = cfg.world_units_per_cell as f32;
    w.params.death_probability = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.spawn_energy = 10.0;
    // N (AAU) + L (CUU) = quimiossíntese variante 0 (produtor).
    let text = format!("AUG {} {} UAA", "AAU CUU GGU ".repeat(3), "GGU ".repeat(6));
    let genome: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    let dists = [5.0f32, 15.0, 30.0, 60.0, 120.0];
    let mut reqs = Vec::new();
    for &d in &dists {
        for i in 0..60 {
            let ang = i as f32 / 60.0 * std::f32::consts::TAU;
            let (x, y) = ((hx + d * ang.cos()) * wpc, (hy + d * ang.sin()) * wpc);
            reqs.push(SpawnRequest::with_genome(x, y, &genome));
        }
    }
    w.request_seeds(&reqs);
    run(&mut w, 64);
    let a0 = w.read_agents_blocking(&gpu);
    let start: std::collections::HashMap<u32, f32> = a0
        .iter()
        .filter(|a| a.alive != 0)
        .map(|a| (a.id, ((a.pos_x / wpc - hx).powi(2) + (a.pos_y / wpc - hy).powi(2)).sqrt()))
        .collect();
    let temp = w.read_f32_blocking(&gpu, &w.temp_buf);
    let redox = w.read_f32_blocking(&gpu, &w.redox_buf);
    let f = cfg.fluid_size as usize;
    let at = |buf: &[f32], d: f32| -> f32 {
        let mut s = 0.0;
        for i in 0..16 {
            let ang = i as f32 / 16.0 * std::f32::consts::TAU;
            let (x, y) = (hx + d * ang.cos(), hy + d * ang.sin());
            let (fx, fy) = ((x / g as f32 * f as f32) as usize, (y / g as f32 * f as f32) as usize);
            s += buf[fy.min(f - 1) * f + fx.min(f - 1)];
        }
        s / 16.0
    };
    run(&mut w, 1500);
    let a1 = w.read_agents_blocking(&gpu);
    println!("fumarola mais quente em ({hx:.0}, {hy:.0}) células; oxidação do redutor {}", w.params.redox_decay);
    for &d in &dists {
        let ids: Vec<u32> = start.iter().filter(|(_, r)| (*r - d).abs() < d * 0.3 + 2.0).map(|(id, _)| *id).collect();
        let alive: Vec<f32> = a1.iter().filter(|a| a.alive != 0 && ids.contains(&a.id)).map(|a| a.energy).collect();
        println!(
            "  a {d:5.0} células: T {:5.2}  redutor {:6.3}  vivos {:2}/{:2}  energia média {:.1}",
            at(&temp, d),
            at(&redox, d),
            alive.len(),
            ids.len(),
            alive.iter().sum::<f32>() / alive.len().max(1) as f32
        );
    }
}
