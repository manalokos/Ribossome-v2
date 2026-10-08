//! O ARMAZENAMENTO FAZ NADAR MAIS DEPRESSA? Nadadores iguais (relógio +
//! glicinas) com e sem um órgão de armazenamento grande (W + W), em água
//! parada sem terreno. Deslocamento líquido ao fim de STEPS passos (média de
//! 64 de cada), com a inércia do mundo e com ela a zero, e nos modos de
//! sinais 0 e 2.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(640);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let gly = |n: usize| "GGU ".repeat(n);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("nadador (relógio + 15 glicinas)", bases(&format!("AUG CAU CUU GAA {} UAA", gly(15)))),
        ("+ armazenamento grande na ponta C", bases(&format!("AUG CAU CUU GAA {} UGG UGG GAA UAA", gly(15)))),
        ("+ armazenamento grande a seguir ao relógio", bases(&format!("AUG CAU CUU GAA UGG UGG GAA {} UAA", gly(15)))),
        ("+ armazenamento pequeno (C M) na ponta C", bases(&format!("AUG CAU CUU GAA {} UGU AUG GAA UAA", gly(15)))),
        ("sem relógio, com armazenamento grande", bases(&format!("AUG {} UGG UGG GAA UAA", gly(16)))),
    ];
    for (mode, inertia_on) in [(0.0f32, true), (0.0, false), (2.0, true), (2.0, false)] {
        let mut w = World::new(&gpu, cfg, 3);
        w.custom_terrain = Some((vec![0; cfg.cells() as usize], vec![0.0; cfg.cells() as usize]));
        w.fumaroles.clear();
        w.seed_matter(&gpu, 3);
        w.settings.fluid_enabled = false;
        w.settings.contact_enabled = false;
        w.params.death_probability = 0.0;
        w.params.maintenance_cost = 0.0;
        w.params.pairing_rate = 0.0;
        w.params.uptake_rate = 0.0;
        w.params.uv_strength = 0.0;
        w.params.sedimentation = 0.0;
        w.params.brownian_rot = 0.0;
        w.params.spawn_energy = 60.0;
        w.params.signal_mode = mode;
        let inertia = w.params.inertia;
        if !inertia_on {
            w.params.inertia = 0.0;
        }
        let reps = 64;
        let mut reqs = Vec::new();
        for (ci, (_, g)) in cases.iter().enumerate() {
            for r in 0..reps {
                reqs.push(SpawnRequest::with_genome(4000.0 + 800.0 * r as f32, 4000.0 + 6000.0 * ci as f32, g));
            }
        }
        w.request_seeds(&reqs);
        let run = |w: &mut World, n: u32| {
            let mut left = n;
            while left > 0 {
                let k = left.min(64);
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                w.encode_steps(&gpu.queue, &mut enc, k);
                gpu.queue.submit([enc.finish()]);
                gpu.wait_idle();
                left -= k;
            }
        };
        run(&mut w, 2);
        let a0 = w.read_agents_blocking(&gpu);
        run(&mut w, steps);
        let a1 = w.read_agents_blocking(&gpu);
        println!("modo dos sinais {mode}, inércia {} — deslocamento líquido em {steps} passos (unidades do mundo)", if inertia_on { format!("{inertia}") } else { "0".into() });
        for (ci, (name, _)) in cases.iter().enumerate() {
            let y = 4000.0 + 6000.0 * ci as f32;
            let (mut sum, mut n, mut best) = (0.0f32, 0u32, 0.0f32);
            for a in a0.iter().filter(|a| a.alive != 0 && (a.pos_y - y).abs() < 400.0) {
                if let Some(b) = a1.iter().find(|b| b.alive != 0 && b.id == a.id) {
                    let d = ((b.pos_x - a.pos_x).powi(2) + (b.pos_y - a.pos_y).powi(2)).sqrt();
                    sum += d;
                    best = best.max(d);
                    n += 1;
                }
            }
            println!("  {name:44} média {:8.1}  máximo {:8.1}  (n={n})", sum / n.max(1) as f32, best);
        }
    }
}
