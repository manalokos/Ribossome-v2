//! As duas bandas de luz de uma cena (a de cima e a funda): média por faixas
//! de altura, do topo ao fundo, depois de STEPS passos. Só lê a cena.
//! SCENE=caminho, STEPS=n, PARAMS=nome=valor,...
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::shaders::LIGHT_DIV;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let gpu = Gpu::new_headless().unwrap();
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    let cfg = {
        let c = &scene.header["cfg"];
        let g = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        WorldConfig { grid_size: g("grid_size", 2048), fluid_size: g("fluid_size", 1024), world_units_per_cell: g("world_units_per_cell", 30), max_agents: g("max_agents", 400_000) }
    };
    let ls = (cfg.grid_size / LIGHT_DIV) as usize;
    let mut w = World::new(&gpu, cfg, 1);
    w.load_scene(&gpu, &scene).unwrap();
    if let Ok(list) = std::env::var("PARAMS") {
        for kv in list.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("PARAMS: nome=valor");
            assert!(w.params.set_named(k, v.parse().expect("valor")), "parâmetro desconhecido: {k}");
        }
    }
    let mut k = 0;
    while k < steps {
        let s = (steps - k).min(MAX_STEPS_PER_FRAME);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, s);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        k += s;
    }
    let light = w.read_f32_blocking(&gpu, &w.light_buf);
    println!("epoch {}, sol {:.2}, uv_depth {}", w.params.epoch, w.params.daylight(w.params.epoch), w.params.uv_depth);
    println!("altura (topo = 100%)   luz de cima   luz funda   funda × 0,4 (o que rende)");
    let bands = 10;
    for b in 0..bands {
        let (y1, y0) = (ls - b * ls / bands, ls - (b + 1) * ls / bands);
        let mean = |off: usize| (y0..y1).flat_map(|y| (0..ls).map(move |x| off + y * ls + x)).map(|i| light[i] as f64).sum::<f64>() / ((y1 - y0) * ls) as f64;
        let (top, deep) = (mean(0), mean(ls * ls));
        println!("  {:>3}–{:>3}%   {top:>10.4}   {deep:>9.4}   {:>9.4}", 100 * y0 / ls, 100 * y1 / ls, deep * 0.4);
    }
}
