//! Proteases com alcance, família pelo vizinho, imunidade e custo de estar
//! ligada. Pares caçador + presa a quatro distâncias (a tocar, a ~25, ~75 e
//! ~135 unidades de folga), sem nada que os mexa a não ser o contacto, e a
//! energia que a presa perde em STEPS passos. Casos:
//!   contacto + vizinho E (corta lisina) contra presa de lisina;
//!   alcance 40 + E; alcance 100 por δ + E (desligada: sem sinal);
//!   contacto + vizinho K (corta aspartato) contra presa de lisina: nada;
//!   contacto + E contra outro caçador igual: imunes um ao outro.
//! No fim, quanto gasta por passo um caçador sozinho de cada tipo.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    // W (UGG) + modificador + GAA (intensidade): R = variante 0 (contacto,
    // sempre), G = 2 (alcance 40, sempre), D = 5 (alcance 100, por δ).
    let hunter = |variant: &str, antenna: &str| bases(&format!("AUG UGG {variant} GAA {antenna} GGU GGU GGU GGU UAA"));
    let lys = bases(&format!("AUG {} UAA", "AAA ".repeat(10)));
    let gly = bases(&format!("AUG {} UAA", "GGU ".repeat(10)));
    // Presas de lisina com um INIBIDOR (N + M): vizinho E = família 1 (a do caçador), vizinho K = família 2.
    let lys_inib1 = bases(&format!("AUG AAU AUG GAA GAA {} UAA", "AAA ".repeat(10)));
    let lys_inib2 = bases(&format!("AUG AAU AUG GAA {} UAA", "AAA ".repeat(10)));
    let cases: [(&str, Vec<u8>, Vec<u8>); 10] = [
        ("contacto, corta lisina / presa de lisina", hunter("CGU", "GAA"), lys.clone()),
        ("alcance 40, corta lisina / presa de lisina", hunter("GGU", "GAA"), lys.clone()),
        ("alcance 100 por δ (sem sinal) / presa de lisina", hunter("GAU", "GAA"), lys.clone()),
        ("contacto, corta aspartato / presa de lisina", hunter("CGU", "AAA"), lys.clone()),
        ("contacto, corta lisina / outro caçador igual", hunter("CGU", "GAA"), hunter("CGU", "GAA")),
        ("contacto, corta lisina / presa de glicina", hunter("CGU", "GAA"), gly.clone()),
        ("contacto, GENERALISTA / presa de glicina", hunter("CGU", "GGU"), gly.clone()),
        ("contacto, GENERALISTA / presa de lisina", hunter("CGU", "GGU"), lys.clone()),
        ("contacto, corta lisina / lisina + INIBIDOR da família 1", hunter("CGU", "GAA"), lys_inib1.clone()),
        ("contacto, corta lisina / lisina + inibidor da família 2", hunter("CGU", "GAA"), lys_inib2.clone()),
    ];
    let gaps = [-10.0f32, 25.0, 75.0, 135.0];
    let reps = 16;
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
    // Primeiro, um de cada sozinho, para saber os raios de contacto.
    let s = cfg.sim_size();
    let mut reqs = Vec::new();
    for (ci, (_, h, p)) in cases.iter().enumerate() {
        reqs.push(SpawnRequest::with_genome(1000.0 + 800.0 * ci as f32, s - 1000.0, h));
        reqs.push(SpawnRequest::with_genome(1000.0 + 800.0 * ci as f32, s - 2000.0, p));
    }
    w.request_seeds(&reqs);
    let run = |w: &mut World, k: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    run(&mut w, 2);
    let agents = w.read_agents_blocking(&gpu);
    let near = |x: f32, y: f32| agents.iter().find(|a| a.alive != 0 && (a.pos_x - x).abs() < 200.0 && (a.pos_y - y).abs() < 200.0).map(|a| a.radius).unwrap_or(20.0);
    let radii: Vec<(f32, f32)> = (0..cases.len()).map(|ci| (near(1000.0 + 800.0 * ci as f32, s - 1000.0), near(1000.0 + 800.0 * ci as f32, s - 2000.0))).collect();
    // Agora os pares, em grelha: caso × folga × repetições.
    let mut reqs = Vec::new();
    let mut posts: Vec<(usize, usize, f32, f32)> = Vec::new();
    for (ci, (_, h, p)) in cases.iter().enumerate() {
        for (gi, gap) in gaps.iter().enumerate() {
            for r in 0..reps {
                let x = 1500.0 + 700.0 * (gi * reps + r) as f32;
                let y = 2000.0 + 1500.0 * ci as f32;
                let d = radii[ci].0 + radii[ci].1 + gap;
                reqs.push(SpawnRequest::with_genome(x, y, h));
                reqs.push(SpawnRequest::with_genome(x + d, y, p));
                posts.push((ci, gi, x, y));
            }
        }
    }
    w.request_seeds(&reqs);
    run(&mut w, 1);
    let before = w.read_agents_blocking(&gpu);
    run(&mut w, steps);
    let after = w.read_agents_blocking(&gpu);
    let energy = |list: &[ribossome::params::Agent], id: u32| list.iter().find(|a| a.alive != 0 && a.id == id).map(|a| a.energy);
    // (perda da presa, perda do caçador) por caso × folga.
    let mut sum = vec![[(0.0f32, 0.0f32, 0u32); 4]; cases.len()];
    for &(ci, gi, x, y) in &posts {
        let d = radii[ci].0 + radii[ci].1 + gaps[gi];
        let find = |px: f32| before.iter().filter(|a| a.alive != 0 && (a.pos_y - y).abs() < 100.0).min_by(|a, b| (a.pos_x - px).abs().total_cmp(&(b.pos_x - px).abs()));
        if let (Some(h), Some(p)) = (find(x), find(x + d)) {
            let lost_p = p.energy - energy(&after, p.id).unwrap_or(0.0);
            let lost_h = h.energy - energy(&after, h.id).unwrap_or(0.0);
            sum[ci][gi].0 += lost_p;
            sum[ci][gi].1 += lost_h;
            sum[ci][gi].2 += 1;
        }
    }
    println!("energia perdida pela PRESA em {steps} passos (média de {reps} pares), por folga entre os corpos");
    println!("{:50} {:>9} {:>9} {:>9} {:>9}", "caso", "a tocar", "folga 25", "folga 75", "folga 135");
    for (ci, (name, _, _)) in cases.iter().enumerate() {
        let v: Vec<String> = sum[ci].iter().map(|(p, _, n)| format!("{:9.2}", p / (*n).max(1) as f32)).collect();
        println!("{name:50} {}", v.join(" "));
    }
    println!("energia do CAÇADOR (perda; negativo = ganhou), mesmas colunas");
    for (ci, (name, _, _)) in cases.iter().enumerate() {
        let v: Vec<String> = sum[ci].iter().map(|(_, h, n)| format!("{:9.2}", h / (*n).max(1) as f32)).collect();
        println!("{name:50} {}", v.join(" "));
    }
    // Custo de estar ligada: caçadores sozinhos, com manutenção normal.
    w.params.maintenance_cost = 0.002;
    let alone: Vec<(&str, Vec<u8>)> = vec![
        ("sem protease (7 glicinas)", bases("AUG GGU GGU GGU GGU GGU GGU UAA")),
        ("contacto, sempre ligada", hunter("CGU", "GAA")),
        ("alcance 40, sempre ligada", hunter("GGU", "GAA")),
        ("alcance 100 por δ, desligada", hunter("GAU", "GAA")),
    ];
    let reqs: Vec<SpawnRequest> = alone.iter().enumerate().map(|(i, (_, g))| SpawnRequest::with_genome(3000.0 + 2000.0 * i as f32, s - 4000.0, g)).collect();
    w.request_seeds(&reqs);
    run(&mut w, 1);
    let b0 = w.read_agents_blocking(&gpu);
    run(&mut w, 100);
    let b1 = w.read_agents_blocking(&gpu);
    // OUT=prefixo: retratos dos quatro (para ver os espigões das de alcance).
    if let Ok(out) = std::env::var("OUT") {
        let cap = ribossome::render::capture::Capture::new(&gpu, &w, 384);
        for (i, _) in alone.iter().enumerate() {
            let x = 3000.0 + 2000.0 * i as f32;
            if let Some((slot, a)) = b1.iter().enumerate().find(|(_, a)| a.alive != 0 && (a.pos_x - x).abs() < 300.0 && (a.pos_y - (s - 4000.0)).abs() < 300.0) {
                cap.view.focus.set(slot as u32);
                cap.view.focus_offset.set([0.0, 0.0]);
                let cam = ribossome::render::Camera { center: [a.pos_x, a.pos_y], zoom: 384.0 / 330.0 };
                let rgba = cap.render(&gpu, &w, &cam, 0, 0.1);
                cap.save_png(&rgba, std::path::Path::new(&format!("{out}_{i}.png"))).unwrap();
            }
        }
    }
    println!("gasto por passo de um agente sozinho (manutenção 0,002):");
    for (i, (name, _)) in alone.iter().enumerate() {
        let x = 3000.0 + 2000.0 * i as f32;
        if let Some(a) = b0.iter().find(|a| a.alive != 0 && (a.pos_x - x).abs() < 300.0 && (a.pos_y - (s - 4000.0)).abs() < 300.0) {
            let e1 = energy(&b1, a.id).unwrap_or(0.0);
            println!("  {name:32} {:.5}", (a.energy - e1) / 100.0);
        }
    }
}
