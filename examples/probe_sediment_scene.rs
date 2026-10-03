//! Sedimentos numa cena real: carrega uma cena (por omissão o autosave),
//! corre STEPS passos e mede quanto entulho se mexeu: células cujo número de
//! grãos mudou, grãos que mudaram de sítio (metade da soma das diferenças) e
//! a subida média do centro de massa do entulho. Grava a vista do terreno
//! antes e depois (target/snapshots/sedimento_*.png). Variáveis: SCENE,
//! STEPS, TRANSPORT, THRESHOLD, SETTLE (por cima dos da cena), VIEW (vista
//! das imagens, 6 = terreno).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{MAX_STEPS_PER_FRAME, Scene, World};

fn main() {
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 1);
    let scene = Scene::read(std::path::Path::new(&path)).unwrap();
    w.load_scene(&gpu, &scene).unwrap();
    if let Some(v) = envf("TRANSPORT") {
        w.params.sediment_transport = v;
    }
    if let Some(v) = envf("THRESHOLD") {
        w.params.sediment_threshold = v;
    }
    if let Some(v) = envf("SETTLE") {
        w.params.sediment_settle = v;
    }
    println!(
        "arrasto {} crítica {} queda {}",
        w.params.sediment_transport, w.params.sediment_threshold, w.params.sediment_settle
    );
    let g = cfg.grid_size as usize;
    let out = std::path::PathBuf::from("target/snapshots");
    std::fs::create_dir_all(&out).unwrap();
    let cap = Capture::new(&gpu, &w, 1024);
    let s = cfg.sim_size();
    let cam = Camera { center: [0.5 * s, 0.5 * s], zoom: 1024.0 / s };
    let shot = |w: &World, name: &str| {
        let rgba = cap.render(&gpu, w, &cam, envf("VIEW").unwrap_or(6.0) as u32, 0.5);
        cap.save_png(&rgba, &out.join(format!("sedimento_{name}.png"))).unwrap();
    };
    // Velocidades do fluido (células do fluido/s): para escolher a crítica.
    let vel = w.read_f32_blocking(&gpu, &w.velocity_buf);
    let mut sp: Vec<f32> = vel.chunks(2).map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt()).collect();
    sp.sort_by(|a, b| a.total_cmp(b));
    let q = |f: f32| sp[((sp.len() - 1) as f32 * f) as usize];
    let up = vel.chunks(2).map(|v| v[1]).fold(0.0f32, f32::max);
    println!("corrente: mediana {:.2}, 90% {:.2}, 99% {:.2}, 99,9% {:.2}, máx {:.2}; subida máx {up:.2}", q(0.5), q(0.9), q(0.99), q(0.999), q(1.0));
    let g0 = w.read_gamma_blocking(&gpu);
    shot(&w, "antes");
    let steps = envf("STEPS").unwrap_or(3000.0) as u32;
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let g1 = w.read_gamma_blocking(&gpu);
    shot(&w, "depois");
    // Mapa da diferença: vermelho = perdeu grãos, verde = ganhou, cinzento =
    // entulho que ficou (cima = cima do mundo).
    let mut rgb = vec![0u8; g * g * 3];
    for y in 0..g {
        for x in 0..g {
            let i = y * g + x;
            let o = ((g - 1 - y) * g + x) * 3;
            let d = g1[i] as i32 - g0[i] as i32;
            if d < 0 {
                rgb[o] = 255;
            } else if d > 0 {
                rgb[o + 1] = 255;
            } else if g0[i] > 0 {
                let v = (40 + 30 * g0[i].min(6)) as u8;
                rgb[o..o + 3].copy_from_slice(&[v, v, v]);
            }
        }
    }
    let file = std::fs::File::create(out.join("sedimento_diferenca.png")).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), g as u32, g as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&rgb).unwrap();
    let changed = g0.iter().zip(&g1).filter(|(a, b)| a != b).count();
    let moved: u64 = g0.iter().zip(&g1).map(|(&a, &b)| (a as i64 - b as i64).unsigned_abs()).sum::<u64>() / 2;
    let com = |gg: &[u32]| {
        let (mut m, mut my) = (0f64, 0f64);
        for (i, &c) in gg.iter().enumerate() {
            m += c as f64;
            my += c as f64 * (i / g) as f64;
        }
        (m, my / m.max(1.0))
    };
    let (m0, y0) = com(&g0);
    let (m1, y1) = com(&g1);
    println!(
        "{steps} passos: {changed} células mudaram, ~{moved} grãos mudaram de sítio; grãos {m0} -> {m1}; centro de massa do entulho {y0:.2} -> {y1:.2} células (subida {:+.3})",
        y1 - y0
    );
}
