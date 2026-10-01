//! Galeria de organismos: corre o mundo, escolhe agentes vivos e grava um
//! PNG com 16 deles ampliados (4×4), centrados no centro de massa.
//! Variáveis: SEEDS, STEPS, MINLEN (resíduos mínimos), OUT, ZOOMPX (píxeis
//! por unidade do mundo).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    world.seed_matter(&gpu, 1);
    let mut rng = ribossome::life::SplitMix(7);
    let reqs = ribossome::life::seed_requests(env("SEEDS", 3000), [30, 200], true, cfg.sim_size(), &mut rng);
    world.request_seeds(&reqs);
    let steps: u32 = env("STEPS", 300);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
    }
    let min_len: u32 = env("MINLEN", 15);
    let agents = world.read_agents_blocking(&gpu);
    let picked: Vec<_> = agents.iter().filter(|a| a.alive != 0 && a.body_len >= min_len).take(16).collect();
    println!(
        "{} vivos com >= {min_len} resíduos; a desenhar {}",
        agents.iter().filter(|a| a.alive != 0 && a.body_len >= min_len).count(),
        picked.len()
    );

    let tile = 256u32;
    let cap = Capture::new(&gpu, &world, tile);
    let zoom: f32 = env("ZOOMPX", 1.0);
    let mut sheet = vec![0u8; (tile * 4 * tile * 4 * 4) as usize];
    for (i, a) in picked.iter().enumerate() {
        let cam = Camera { center: [a.pos_x, a.pos_y], zoom };
        let rgba = cap.render(&gpu, &world, &cam, 0, 0.25);
        let (tx, ty) = ((i % 4) as u32, (i / 4) as u32);
        for y in 0..tile {
            let src = (y * tile * 4) as usize;
            let dst = (((ty * tile + y) * tile * 4 + tx * tile) * 4) as usize;
            sheet[dst..dst + (tile * 4) as usize].copy_from_slice(&rgba[src..src + (tile * 4) as usize]);
        }
        println!("  {i:2}: {} resíduos, geração {}, idade {}", a.body_len, a.generation, a.age);
    }
    let out = std::path::PathBuf::from(env("OUT", String::from("target/gallery.png")));
    let file = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let mut enc = png::Encoder::new(file, tile * 4, tile * 4);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&sheet).unwrap();
    println!("{}", out.display());
}
