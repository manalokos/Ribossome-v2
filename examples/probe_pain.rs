//! DOR e CHEIRO A PROTEASE. Uma presa com dois sensores seguidos de prolina
//! (a antena): um de energia (passa a sentir a mordida) e um de corpos (passa
//! a cheirar proteases abertas). Três situações, 16 de cada:
//!   presa sozinha; presa encostada a um caçador de contacto (sempre
//!   ligado); presa a ~100 unidades de um caçador de alcance 40 (não lhe
//!   chega, mas está dentro do raio do sensor).
//! Mostra o sinal α no resíduo de cada sensor ao fim de STEPS passos.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    // Q + C = sensor de energia (variante 3, β), Q + L = sensor de comida
    // (variante 0, α... o alvo "corpos" é da variante 3: Q + D). Cada órgão:
    // prefixo, modificador, intensidade (GAA), e a seguir a antena (CCU = P).
    // Resíduos: 0 M, 1 sensor de energia (Q C), 2 P, 3 sensor de corpos (Q D), 4 P, depois lisinas.
    let prey = bases(&format!("AUG CAA UGU GAA CCU CAA GAU GAA CCU {} UAA", "AAA ".repeat(8)));
    let plain = bases(&format!("AUG CAA UGU GAA GGU CAA GAU GAA GGU {} UAA", "AAA ".repeat(8)));
    let hunter = |variant: &str| bases(&format!("AUG UGG {variant} GAA GAA GGU GGU GGU GGU UAA"));
    let cases: [(&str, &Vec<u8>, Option<(Vec<u8>, f32)>); 5] = [
        ("presa (antena P) sozinha", &prey, None),
        ("presa (antena P) encostada a caçador de contacto", &prey, Some((hunter("CGU"), -10.0))),
        ("presa (antena P) a 100 de um caçador de alcance 40", &prey, Some((hunter("GGU"), 100.0))),
        ("presa (antena G) encostada a caçador de contacto", &plain, Some((hunter("CGU"), -10.0))),
        ("presa (antena G) sozinha", &plain, None),
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
    w.params.brownian_rot = 0.0;
    w.params.uv_strength = 0.0;
    w.params.sedimentation = 0.0;
    w.params.signal_crosstalk = 0.0;
    w.params.spawn_energy = 8.0;
    let reps = 16;
    let mut reqs = Vec::new();
    let mut posts = Vec::new();
    for (ci, (_, p, h)) in cases.iter().enumerate() {
        for r in 0..reps {
            let x = 3000.0 + 1500.0 * r as f32;
            let y = 3000.0 + 2500.0 * ci as f32;
            reqs.push(SpawnRequest::with_genome(x, y, p));
            if let Some((hg, gap)) = h {
                reqs.push(SpawnRequest::with_genome(x + 45.0 + gap, y, hg));
            }
            posts.push((ci, x, y));
        }
    }
    w.request_seeds(&reqs);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    w.encode_steps(&gpu.queue, &mut enc, steps);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let agents = w.read_agents_blocking(&gpu);
    let sig: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.signals_buf)).to_vec();
    println!("sinal no resíduo de cada sensor ao fim de {steps} passos (média de {reps}); corpo da presa com 12+ resíduos");
    println!("{:55} {:>10} {:>10} {:>8}", "caso", "dor (γ+δ)", "cheiro (α+β)", "energia");
    for (ci, (name, _, _)) in cases.iter().enumerate() {
        let (mut s1, mut s3, mut e, mut n) = (0.0f32, 0.0f32, 0.0f32, 0);
        for &(c, x, y) in &posts {
            if c != ci {
                continue;
            }
            // A presa é o agente mais comprido perto do sítio.
            let found = agents.iter().enumerate().filter(|(_, a)| a.alive != 0 && a.body_len >= 12 && (a.pos_x - x).abs() < 300.0 && (a.pos_y - y).abs() < 300.0).min_by(|a, b| (a.1.pos_x - x).abs().total_cmp(&(b.1.pos_x - x).abs()));
            if let Some((slot, a)) = found {
                // Os sensores emitem no seu resíduo; soma dos dois canais α e β.
                s1 += sig[slot * 64 + 1][2] + sig[slot * 64 + 1][3];
                s3 += sig[slot * 64 + 3][0] + sig[slot * 64 + 3][1];
                e += a.energy;
                n += 1;
            }
        }
        let n = n.max(1) as f32;
        println!("{name:55} {:10.4} {:10.4} {:8.2}", s1 / n, s3 / n, e / n);
    }
}
