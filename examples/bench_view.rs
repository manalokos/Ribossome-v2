//! Custo do desenho do fundo: tempo por frame (só o desenho, sem ler de
//! volta) de uma imagem SIZE × SIZE da cena, para vários enquadramentos
//! (largura em células) e raios do círculo de confusão. A/B no mesmo
//! processo; mostra o MÍNIMO de 120 repetições de 8 desenhos (a GPU pode estar ocupada).
use std::time::Instant;

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let size: u32 = std::env::var("SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    w.encode_steps(&gpu.queue, &mut enc, 1);
    w.encode_draw_list(&mut enc);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let cap = Capture::new(&gpu, &w, size);
    let s = cfg.sim_size();
    let time = |span: f32, coc: f32| -> f32 {
        cap.view.coc_radius.set(coc);
        let cam = Camera { center: [0.5 * s, 0.5 * s], zoom: size as f32 / (span * cfg.world_units_per_cell as f32) };
        let mut best = f32::MAX;
        for _ in 0..120 {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            // 4 desenhos por medida, para o tempo de submeter pesar menos.
            for _ in 0..8 {
                cap.encode(&gpu.queue, &mut enc, &cam, 0, 0.5);
            }
            let t = Instant::now();
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            best = best.min(t.elapsed().as_secs_f32() * 1000.0 / 8.0);
        }
        best
    };
    println!("imagem {size} × {size} ({:.1} milhões de píxeis; um ecrã 1920 × 1080 tem 2,1)", (size * size) as f32 / 1e6);
    for span in [2048.0f32, 400.0, 120.0, 40.0, 12.0] {
        let base = time(span, 0.0);
        let line: Vec<String> = [0.15f32, 0.35, 0.8].iter().map(|&c| format!("raio {c}: {:.2} ms", time(span, c))).collect();
        println!("  {span:6.0} células de largura (píxel = {:.3} células): quadrados {base:.2} ms; {}", span / size as f32, line.join("; "));
    }
}
