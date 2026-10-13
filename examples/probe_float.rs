//! O flutuador: corpos iguais (10 glicinas) sem flutuador, com um fraco (H+F,
//! impulsão 2), com um forte (H+N, impulsão 5) e com dois fortes, a meia
//! altura de um mundo vazio com gravidade nos agentes. Quanto sobem ou descem
//! em STEPS passos (média de 32 de cada; + = para cima).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let gly = "GGU ".repeat(10);
    let cases: [(&str, Vec<u8>); 4] = [
        ("sem flutuador", bases(&format!("AUG {gly} UAA"))),
        ("um flutuador fraco (impulsão 2)", bases(&format!("AUG CAU UUU GAA {gly} UAA"))),
        ("um flutuador forte (impulsão 5)", bases(&format!("AUG CAU AAU GAA {gly} UAA"))),
        ("dois flutuadores fortes", bases(&format!("AUG CAU AAU GAA CAU AAU GAA {gly} UAA"))),
    ];
    let mut w = World::new(&gpu, cfg, 3);
    w.custom_terrain = Some((vec![0; cfg.cells() as usize], vec![0.0; cfg.cells() as usize]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.uptake_rate = 0.0;
    w.params.uv_strength = 0.0;
    w.params.photo_damage = 0.0;
    w.params.naked_decay = 0.0;
    w.params.sedimentation = 0.05;
    w.params.spawn_energy = 8.0;
    let s = cfg.sim_size();
    let reps = 32usize;
    let mut reqs = Vec::new();
    // (Cada caso na sua altura, para os distinguir depois.)
    let row = |c: usize| s * (0.35 + 0.1 * c as f32);
    for (c, (_, g)) in cases.iter().enumerate() {
        for r in 0..reps {
            reqs.push(SpawnRequest::with_genome(s * (0.1 + 0.8 * r as f32 / reps as f32), row(c), g));
        }
    }
    w.request_seeds(&reqs);
    let run = |w: &mut World, n: u32| {
        let mut k = 0;
        while k < n {
            let st = (n - k).min(MAX_STEPS_PER_FRAME);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, st);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            k += st;
        }
    };
    run(&mut w, 2);
    let before = w.read_agents_blocking(&gpu);
    run(&mut w, steps);
    let after = w.read_agents_blocking(&gpu);
    println!("gravidade nos agentes {}, {steps} passos (mundo de {s:.0} unidades)", w.params.sedimentation);
    for (c, (name, _)) in cases.iter().enumerate() {
        let (mut n, mut dy, mut body) = (0u32, 0f64, 0u32);
        for (a, b) in before.iter().zip(&after) {
            if a.alive != 0 && b.alive != 0 && a.id == b.id && (a.pos_y - row(c)).abs() < s * 0.02 {
                n += 1;
                dy += (b.pos_y - a.pos_y) as f64;
                body = b.body_len;
            }
        }
        println!("  {name:<34} {n:>3} vivos, {body:>2} resíduos: {:+9.1} unidades ({:+.4} por passo)", dy / n.max(1) as f64, dy / n.max(1) as f64 / steps as f64);
    }
}
