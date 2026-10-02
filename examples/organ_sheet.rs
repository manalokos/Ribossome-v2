//! Folha de órgãos: um organismo construído por tipo de órgão (o órgão a
//! meio de 12 alaninas, com um relógio no início), desenhados ampliados.
//! Linha de cima: cor química; linha de baixo: a mesma cena com os sinais
//! α (vermelho) e β (verde). OUT = caminho do PNG.
use ribossome::gpu::Gpu;
use ribossome::life::organs::{ORGAN_NAMES, ORGAN_TYPES};
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::render::Camera;
use ribossome::render::capture::Capture;
use ribossome::world::World;

const BASES: [char; 4] = ['A', 'U', 'G', 'C'];

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

/// Codão cujo índice (A U G C, base 4) é `idx`.
fn codon_of(idx: usize) -> String {
    [BASES[idx / 16], BASES[(idx / 4) % 4], BASES[idx % 4]].iter().collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { grid_size: 1024, fluid_size: 512, max_agents: 1024, ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 5);
    world.configure_lab();
    world.seed_lab(&gpu, 5, 1.5);
    world.params.pairing_rate = 0.0;
    world.params.death_probability = 0.0;
    world.params.maintenance_cost = 0.0;
    world.params.sedimentation = 0.0;
    let s = cfg.sim_size();
    let ala = "GCU ".repeat(6);
    let mut reqs = Vec::new();
    for t in 0..ORGAN_TYPES {
        // Promotor UGU (C) + modificador t (parâmetro 0) + intensidade por omissão; relógio no início.
        let g = format!("AUG UGU UCU GGU {ala} UGU {} UUC {ala} UAA", codon_of(t));
        let x = s * (0.1 + 0.08 * t as f32);
        reqs.push(SpawnRequest::with_genome(x, s * 0.5, &bases(&g)));
    }
    world.request_seeds(&reqs);
    for _ in 0..10 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, 30);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    }
    let mut agents: Vec<_> = world.read_agents_blocking(&gpu).into_iter().filter(|a| a.alive != 0).collect();
    agents.sort_by(|a, b| a.pos_x.partial_cmp(&b.pos_x).unwrap());
    let tile = 256u32;
    let cols = agents.len() as u32;
    let cap = Capture::new(&gpu, &world, tile);
    let mut sheet = vec![0u8; (tile * cols * tile * 2 * 4) as usize];
    for (row, sv) in [0u32, 3].iter().enumerate() {
        cap.signal_view.set(*sv);
        for (i, a) in agents.iter().enumerate() {
            let cam = Camera { center: [a.pos_x, a.pos_y], zoom: 1.2 };
            let rgba = cap.render(&gpu, &world, &cam, 0, 0.15);
            for y in 0..tile {
                let src = (y * tile * 4) as usize;
                let dst = ((((row as u32) * tile + y) * tile * cols + i as u32 * tile) * 4) as usize;
                sheet[dst..dst + (tile * 4) as usize].copy_from_slice(&rgba[src..src + (tile * 4) as usize]);
            }
        }
    }
    for (i, n) in ORGAN_NAMES.iter().enumerate() {
        println!("coluna {i}: {n}");
    }
    let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or("target/organs.png".into()));
    let file = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    let mut enc = png::Encoder::new(file, tile * cols, tile * 2);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&sheet).unwrap();
    println!("{} ({} agentes)", out.display(), agents.len());
}
