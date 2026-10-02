//! Trajetória de um nadador construído (relógio + glicinas) no laboratório:
//! posição e orientação a cada 20 passos, caminho percorrido, deslocamento
//! líquido e rotação acumulada.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { grid_size: 1024, fluid_size: 512, max_agents: 1024, ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 3);
    world.configure_lab();
    world.seed_lab(&gpu, 3, 6.0);
    world.params.pairing_rate = 0.0;
    world.params.death_probability = 0.0;
    world.params.swim_gain = std::env::var("SWIM").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
    // relógio sem segundo modificador (UAA... não: espaçador AUG não é stop) -> usa GCU GCU como intensidade.
    let genome = std::env::var("GENOME").unwrap_or_else(|_| format!("AUG UGU UCU {} UAA", "GGU ".repeat(15)));
    let s = cfg.sim_size();
    world.request_seeds(&[SpawnRequest::with_genome(s * 0.5, s * 0.5, &bases(&genome))]);
    let mut last: Option<(f32, f32)> = None;
    let (mut path, mut rot0, mut start) = (0.0f32, 0.0f32, (0.0f32, 0.0f32));
    for t in 0..=60 {
        {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            world.encode_steps(&gpu.queue, &mut enc, if t == 0 { 1 } else { 20 });
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
        }
        let a = *world.read_agents_blocking(&gpu).iter().find(|a| a.alive != 0).expect("vivo");
        if t == 0 {
            start = (a.pos_x, a.pos_y);
            rot0 = a.rot;
        }
        if let Some((x, y)) = last {
            path += (a.pos_x - x).hypot(a.pos_y - y);
        }
        last = Some((a.pos_x, a.pos_y));
        if t % 6 == 0 {
            println!(
                "passo {:>5}: pos ({:8.1}, {:8.1})  rotação acumulada {:7.2} rad  ({} resíduos)",
                t * 20,
                a.pos_x - start.0,
                a.pos_y - start.1,
                a.rot - rot0,
                a.body_len
            );
        }
    }
    // Rotação passo a passo durante 160 passos: oscilação vs deriva.
    let mut rots = Vec::new();
    for _ in 0..160 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, 1);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        rots.push(world.read_agents_blocking(&gpu).iter().find(|a| a.alive != 0).unwrap().rot);
    }
    let d: Vec<f32> = rots.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = d.iter().sum::<f32>() / d.len() as f32;
    let (mn, mx) = d.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    println!("Ω por passo: média {mean:.4}, mín {mn:.4}, máx {mx:.4}");
    let line: Vec<String> = d.iter().step_by(4).map(|v| format!("{v:+.3}")).collect();
    println!("{}", line.join(" "));
    let a = *world.read_agents_blocking(&gpu).iter().find(|a| a.alive != 0).unwrap();
    let net = (a.pos_x - start.0).hypot(a.pos_y - start.1);
    println!("caminho {path:.1}, deslocamento líquido {net:.1}, rotação total {:.2} rad", a.rot - rot0);
}
