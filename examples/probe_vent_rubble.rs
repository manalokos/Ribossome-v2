//! A FUMAROLA LIMPA A SAÍDA? Mundo vazio com um chão de rocha, uma fumarola
//! pousada nele e um monte de entulho por cima da saída. Conta os grãos
//! soltos dentro do disco da fumarola ao longo do tempo. BLOW=0 não se pode
//! desligar aqui (é uma constante do shader): compara-se com um monte igual
//! ao lado, sem fumarola.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::World;

fn main() {
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { grid_size: 512, fluid_size: 256, max_agents: 4096, ..WorldConfig::DEFAULT };
    let mut w = World::new(&gpu, cfg, 1);
    w.use_empty_terrain();
    // SETTLE = gravidade dos grãos (por omissão a do código); COMPACT = compactação.
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    w.params.sediment_settle = envf("SETTLE", w.params.sediment_settle);
    w.params.sediment_compaction = envf("COMPACT", w.params.sediment_compaction);
    println!("gravidade dos grãos {}, compactação {}", w.params.sediment_settle, w.params.sediment_compaction);
    w.seed_matter(&gpu, 1);
    let n = cfg.grid_size as f32;
    let (floor, r) = (40.0, 10.0);
    let (vent_x, plain_x) = (n * 0.3, n * 0.7);
    // Cada pincelada vai no seu envio (os parâmetros do pincel são um só
    // bloco, escrito antes de os comandos correrem).
    let paint = |w: &mut World, cx: f32, cy: f32, radius: f32, grains: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_paint(&gpu.queue, &mut enc, cx, cy, radius, grains);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    // Chão de rocha a toda a largura (discos encostados), e dois montes de entulho.
    let mut x = 0.0;
    while x < n {
        paint(&mut w, x, floor - 30.0, 30.0, 6);
        x += 20.0;
    }
    for cx in [vent_x, plain_x] {
        paint(&mut w, cx, floor + r, r, 2);
    }
    w.paint_source(vent_x, floor + r, r, Some(1.0), Some(1.0));
    let count = |w: &World, cx: f32| {
        let g = w.read_gamma_blocking(&gpu);
        let mut c = 0u32;
        for y in 0..cfg.grid_size {
            for xx in 0..cfg.grid_size {
                let (dx, dy) = (xx as f32 - cx, y as f32 - (floor + r));
                let v = g[(y * cfg.grid_size + xx) as usize];
                if dx * dx + dy * dy <= r * r && v > 0 && v < 3 {
                    c += v;
                }
            }
        }
        c
    };
    // Grãos soltos bem acima do chão (em suspensão, a espalharem-se pelo mundo).
    let afloat = |w: &World| {
        let g = w.read_gamma_blocking(&gpu);
        let y0 = (floor + 4.0 * r) as u32;
        (y0..cfg.grid_size).flat_map(|y| (0..cfg.grid_size).map(move |x| (x, y))).map(|(x, y)| g[(y * cfg.grid_size + x) as usize]).filter(|&v| v > 0 && v < 3).sum::<u32>()
    };
    println!("{:>7} {:>22} {:>22} {:>14}", "passos", "grãos sobre a fumarola", "grãos no monte ao lado", "em suspensão");
    let mut done = 0;
    println!("{done:>7} {:>22} {:>22} {:>14}", count(&w, vent_x), count(&w, plain_x), afloat(&w));
    while done < steps {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, 64);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += 64;
        if done % 512 < 64 {
            println!("{done:>7} {:>22} {:>22} {:>14}", count(&w, vent_x), count(&w, plain_x), afloat(&w));
        }
    }
}
