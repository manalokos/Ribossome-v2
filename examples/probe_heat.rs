//! As fumarolas de um terreno carregado aquecem? TERRAIN = PNG (por omissão
//! terreno.png). Mostra o calor carregado, a temperatura máxima e os
//! ativados perto da fonte ao longo do tempo.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::World;

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    let path = std::env::var("TERRAIN").unwrap_or_else(|_| "terreno.png".into());
    if path != "-" {
        let hot = world.load_terrain_png(std::path::Path::new(&path)).unwrap();
        println!("{path}: {hot} células quentes");
    }
    world.seed_matter(&gpu, 1);
    let heat = world.heat_grid();
    println!("calor na grelha: máx {:.3}, soma {:.1}, fumarolas pontuais {}", heat.iter().cloned().fold(0.0, f32::max), heat.iter().sum::<f32>(), world.fumaroles.len());
    let n = cfg.grid_size as usize;
    let src = heat.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let (sx, sy) = (src % n, src / n);
    world.params.uv_strength = std::env::var("UV").ok().and_then(|v| v.parse().ok()).unwrap_or(world.params.uv_strength);
    for round in 0..std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(8) {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, 64);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        let t = world.read_f32_blocking(&gpu, &world.temp_buf);
        let tmax = t.iter().cloned().fold(0.0, f32::max);
        let cells = world.read_cells_blocking(&gpu);
        let frac = |x0: usize, x1: usize, y0: usize, y1: usize| {
            let (mut a, mut sp) = (0u64, 0u64);
            for y in y0..y1.min(n) {
                for x in x0..x1.min(n) {
                    for ch in 0..4 {
                        let v = cells[(y * n + x) * 4 + ch];
                        a += (v & 0xFFFF) as u64;
                        sp += (v >> 16) as u64;
                    }
                }
            }
            a as f64 / (a + sp).max(1) as f64
        };
        let near = frac(sx.saturating_sub(40), sx + 40, sy, sy + 200);
        let far = frac(sx + 300, sx + 380, sy, sy + 200);
        let gam = world.read_gamma_blocking(&gpu);
        let (mut in_rubble, mut rubble_cells, mut in_rock) = (0u64, 0u64, 0u64);
        for i in 0..n * n {
            let tot: u64 = (0..4).map(|ch| { let v = cells[i * 4 + ch]; ((v & 0xFFFF) + (v >> 16)) as u64 }).sum();
            if gam[i] > 0 && gam[i] < 3 { in_rubble += tot; rubble_cells += 1; }
            if gam[i] >= 3 { in_rock += tot; }
        }
        print!("[no entulho {in_rubble} em {rubble_cells} células; na rocha {in_rock}] ");
        let top = frac(0, n, n - 200, n);
        let l = world.read_f32_blocking(&gpu, &world.light_buf);
        let ls = n / ribossome::shaders::LIGHT_DIV as usize;
        print!("[luz topo {:.3} meio {:.4}; ativados no topo {:.1}%] ", l[(ls - 1) * ls + ls / 2], l[(ls / 2) * ls + ls / 2], top * 100.0);
        println!("passo {}: T máx {tmax:.2}  ativados acima da fonte {:.1}%  longe {:.1}%", (round + 1) * 64, near * 100.0, far * 100.0);
    }
}
