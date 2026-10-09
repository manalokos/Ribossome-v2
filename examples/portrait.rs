//! Retratos grandes de agentes de uma cena (para ilustrações): escolhe os N
//! agentes com mais órgãos diferentes (corpo de 14 a 48 resíduos) e desenha
//! cada um sozinho, em fundo preto.
//! SCENE (por omissão o autosave), OUT (prefixo dos PNG), N (3), SIZE (1024),
//! ORGAN (só agentes com este tipo de órgão, ex. 11 = protease).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{Scene, World};

fn main() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let out = std::env::var("OUT").unwrap_or_else(|_| "retrato".into());
    let gpu = Gpu::new_headless().unwrap();
    let mut w = World::new(&gpu, WorldConfig::DEFAULT, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    w.load_scene(&gpu, &scene).unwrap();
    // Uns passos, para os corpos estarem em pose e os fios assentes.
    for _ in 0..3 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, 64);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    }
    let agents = w.read_agents_blocking(&gpu);
    let organs: Vec<u16> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.organs_buf)).to_vec();
    let need: Option<u16> = std::env::var("ORGAN").ok().and_then(|v| v.parse().ok());
    let mut best: Vec<(usize, usize, usize)> = agents
        .iter()
        .enumerate()
        .filter(|(_, a)| a.alive != 0 && (14..=48).contains(&a.body_len))
        .map(|(slot, a)| {
            let o = &organs[slot * 64..slot * 64 + a.body_len as usize];
            let kinds: std::collections::HashSet<u16> = o.iter().filter(|&&c| c != 0).map(|&c| c & 0x1F).collect();
            (kinds.len(), o.iter().filter(|&&c| c != 0).count(), slot, need.is_none_or(|n| kinds.contains(&(n + 1))))
        })
        .filter(|b| b.3)
        .map(|b| (b.0, b.1, b.2))
        .collect();
    best.sort_by(|a, b| b.cmp(a));
    // Um por combinação (tipos, órgãos), para não saírem três iguais.
    best.dedup_by_key(|b| (b.0, b.1));
    let size = env("SIZE", 1024) as u32;
    let cap = Capture::new(&gpu, &w, size);
    let pos: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.body_pos_buf)).to_vec();
    for (i, &(kinds, n, slot)) in best.iter().take(env("N", 3)).enumerate() {
        let a = &agents[slot];
        let (s, c) = a.rot.sin_cos();
        let pts: Vec<[f32; 2]> = pos[slot * 64..slot * 64 + a.body_len as usize].iter().map(|p| [a.pos_x + c * p[0] - s * p[1], a.pos_y + s * p[0] + c * p[1]]).collect();
        let (lo, hi) = pts.iter().fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), p| ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])]));
        let side = ((hi[0] - lo[0]).max(hi[1] - lo[1]) * 1.5 + 120.0).max(160.0);
        let centre = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
        cap.view.focus.set(slot as u32);
        cap.view.focus_offset.set([centre[0] - a.pos_x, centre[1] - a.pos_y]);
        // BRIGHT: brilho dos monómeros (0 = sem eles, para ilustrações limpas).
        let bright: f32 = std::env::var("BRIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let rgba = cap.render(&gpu, &w, &Camera { center: centre, zoom: size as f32 / side }, 0, bright);
        let file = format!("{out}_{i}.png");
        cap.save_png(&rgba, std::path::Path::new(&file)).unwrap();
        println!("{file}: agente {} com {} resíduos, {n} órgãos de {kinds} tipos", a.id, a.body_len);
    }
}
