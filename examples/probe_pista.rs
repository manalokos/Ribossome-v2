//! PISTA DE CORRIDAS (modo de ensaio). Semeia na pista nadadores com um
//! relógio e controlos sem ele, sem mutações, e corre STEPS passos. Mostra:
//!   energia média e vivos de cada tipo (só avançar dá energia);
//!   filhos de cada tipo e se o genoma é IGUAL ao do pai (copy_same);
//!   a matéria por canal antes e depois (tem de ser a mesma);
//!   OUT=prefixo: imagem do mundo (as paredes devem aparecer iluminadas).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn pack(g: &[u8]) -> Vec<u32> {
    let mut w = vec![0u32; 16];
    for (i, &b) in g.iter().enumerate() {
        w[i / 16] |= (b as u32 & 3) << ((i % 16) * 2);
    }
    w
}

fn main() {
    let steps: u32 = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 3);
    w.custom_terrain = Some(ribossome::track::terrain(&cfg));
    w.fumaroles.clear();
    w.seed_density = 0.0;
    w.seed_matter(&gpu, 3);
    w.settings.fluid_enabled = false;
    w.settings.contact_enabled = false;
    ribossome::track::preset(&mut w.params);
    w.params.mutation_rate = 0.0;
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    w.params.track_gain = envf("GAIN", w.params.track_gain);
    w.params.wall_damage = envf("WALL", w.params.wall_damage);
    w.params.signal_mode = envf("MODE", w.params.signal_mode);
    w.params.motor_amplitude = envf("MOTOR", w.params.motor_amplitude);
    println!("ganho {} por unidade, dano das paredes {}", w.params.track_gain, w.params.wall_damage);
    let swimmer = bases(&format!("AUG CAU CUU {} UAA", "GGU ".repeat(15)));
    let control = bases(&format!("AUG {} UAA", "GGU ".repeat(16)));
    let sim = cfg.sim_size();
    let n = 2000;
    let mut reqs = Vec::new();
    let mut rng = 12345u32;
    let mut rnd = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 8) as f32 / 16777216.0
    };
    for i in 0..n {
        let p = ribossome::track::point(sim, rnd(), rnd() * 1.6 - 0.8);
        reqs.push(SpawnRequest::with_genome(p[0], p[1], if i % 2 == 0 { &swimmer } else { &control }));
    }
    w.request_seeds(&reqs);
    let run = |w: &mut World, k: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    run(&mut w, 2);
    let before = w.ledger_blocking(&gpu);
    let per = |l: &ribossome::world::Ledger| -> Vec<u32> { (0..4).map(|c| l.act[c] + l.spent[c] + l.held[c]).collect() };
    let (ps, pc) = (pack(&swimmer), pack(&control));
    let report = |w: &World, label: &str| {
        let agents = w.read_agents_blocking(&gpu);
        let genomes: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
        // [nadador, controlo, outro] × (vivos, energia, filhos)
        let mut t = [(0u32, 0.0f32, 0u32); 3];
        let mut near_wall = [0u32; 3];
        let mut inside_rock = 0;
        for (slot, a) in agents.iter().enumerate() {
            if a.alive == 0 {
                continue;
            }
            let g = &genomes[slot * 16..slot * 16 + 16];
            let kind = if g[..4] == ps[..4] && a.gene_len as usize == swimmer.len() {
                0
            } else if g[..4] == pc[..4] && a.gene_len as usize == control.len() {
                1
            } else {
                2
            };
            t[kind].0 += 1;
            t[kind].1 += a.energy;
            t[kind].2 += (a.generation > 0) as u32;
            near_wall[kind] += (ribossome::track::wall_dist(sim, a.pos_x, a.pos_y) < a.radius.max(1.0)) as u32;
            if ribossome::track::wall_dist(sim, a.pos_x, a.pos_y) < 0.0 {
                inside_rock += 1;
            }
        }
        println!("{label}");
        for (k, name) in ["nadador (relógio)", "controlo", "OUTRO genoma"].iter().enumerate() {
            println!("  {name:20} vivos {:6}  energia média {:6.2}  filhos {:6}  a tocar na parede {:5}", t[k].0, t[k].1 / t[k].0.max(1) as f32, t[k].2, near_wall[k]);
        }
        println!("  com o centro dentro da rocha: {inside_rock}");
    };
    report(&w, "ao fim de 2 passos:");
    let mut done = 2;
    while done < steps {
        let k = 500.min(steps - done);
        run(&mut w, k);
        done += k;
        if done % 500 == 0 || done >= steps {
            report(&w, &format!("ao fim de {done} passos:"));
        }
    }
    let after = w.ledger_blocking(&gpu);
    println!("matéria por canal antes {:?}\n                 depois {:?}  {}", per(&before), per(&after), "(na pista não há monómeros: só conta a matéria dos genomas, que nasce e morre com eles)");
    if let Ok(out) = std::env::var("OUT") {
        let cap = ribossome::render::capture::Capture::new(&gpu, &w, 1024);
        let rgba = cap.render(&gpu, &w, &ribossome::render::Camera { center: [0.5 * sim, 0.5 * sim], zoom: 1024.0 / sim }, 0, 0.5);
        cap.save_png(&rgba, std::path::Path::new(&format!("{out}_mundo.png"))).unwrap();
        let p = ribossome::track::point(sim, 0.1, 0.0);
        let rgba = cap.render(&gpu, &w, &ribossome::render::Camera { center: p, zoom: 1024.0 / (120.0 * 30.0) }, 0, 0.5);
        cap.save_png(&rgba, std::path::Path::new(&format!("{out}_pormenor.png"))).unwrap();
    }
}
