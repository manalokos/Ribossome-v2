//! Há deriva artificial para algum canto? Mundo sem agentes, sem fluido e sem
//! luz, com entulho solto (1 grão) espalhado ao acaso por igual (DENS, fração
//! de células) e monómeros da sementeira. Corre STEPS passos e mede para onde
//! se move o centro de massa do entulho e dos monómeros (em células). Sem
//! nada que escolha uma direção, ambos devem ficar parados (± ruído).
//! TERRAIN=0 desliga os grãos. AGENTS=n semeia n agentes ao acaso por igual
//! com genomas reais de uma cena (SCENE, por omissão o autosave; energia 26,
//! afundamento SEDIM, por omissão 0) e mede também o centro dos agentes.
use ribossome::gpu::Gpu;
use ribossome::life::SplitMix;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn com(v: &[u32], g: usize, f: impl Fn(u32) -> u64) -> (f64, f64, f64) {
    let (mut m, mut mx, mut my) = (0f64, 0f64, 0f64);
    for (i, &c) in v.iter().enumerate() {
        let w = f(c) as f64;
        m += w;
        mx += w * (i % g) as f64;
        my += w * (i / g) as f64;
    }
    (m, mx / m.max(1.0), my / m.max(1.0))
}

fn main() {
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    // Genomas reais (AGENTS > 0): os de agentes vivos de uma cena (por
    // omissão o autosave), lidos antes de criar o mundo de teste.
    let n_agents = envf("AGENTS").unwrap_or(0.0) as u32;
    let mut genomes: Vec<Vec<u8>> = Vec::new();
    if n_agents > 0 {
        let path = std::env::var("SCENE").unwrap_or_else(|_| "saves/autosave.ribo".into());
        let mut src = World::new(&gpu, cfg, 1);
        src.load_scene(&gpu, &ribossome::world::Scene::read(std::path::Path::new(&path)).unwrap()).unwrap();
        let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&src.genomes_buf)).to_vec();
        for (slot, a) in src.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0).take(2000) {
            genomes.push((0..a.gene_len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect());
        }
        println!("{} genomas de {path}", genomes.len());
    }
    let mut w = World::new(&gpu, cfg, 5);
    let g = cfg.grid_size as usize;
    let n = g * g;
    let dens = envf("DENS").unwrap_or(0.1);
    let mut rng = SplitMix(9);
    let gamma: Vec<u32> = (0..n).map(|_| (rng.f32() < dens) as u32).collect();
    w.custom_terrain = Some((gamma, vec![0.0; n]));
    w.fumaroles.clear();
    w.seed_matter(&gpu, 5);
    w.settings.fluid_enabled = false;
    w.settings.terrain_enabled = envf("TERRAIN").unwrap_or(1.0) != 0.0;
    w.params.uv_strength = 0.0;
    w.params.settle = 0.0;
    w.params.sediment_settle = envf("SETTLE").unwrap_or(0.0);
    w.params.sediment_transport = envf("TRANSPORT").unwrap_or(5.0);
    w.params.sediment_threshold = envf("THRESHOLD").unwrap_or(2.5);
    w.params.aggregation = envf("AGG").unwrap_or(1.0);
    w.params.diffusion = envf("DIFF").unwrap_or(4.0);
    w.params.monomer_pressure = 0.0;
    w.params.reactivation_rate = envf("REACT").unwrap_or(3e-5);
    if n_agents > 0 {
        w.params.spawn_energy = 26.0;
        w.params.bioturbation = envf("BIOTURB").unwrap_or(0.14);
        w.params.bioturbation_cost = 0.0;
        w.params.sedimentation = envf("SEDIM").unwrap_or(0.0);
        let mut rr = SplitMix(77);
        let s = cfg.sim_size();
        let reqs: Vec<_> = (0..n_agents)
            .map(|i| {
                let gnm = &genomes[i as usize % genomes.len().max(1)];
                ribossome::params::SpawnRequest::with_genome((0.02 + 0.96 * rr.f32()) * s, (0.02 + 0.96 * rr.f32()) * s, gnm)
            })
            .collect();
        for chunk in reqs.chunks(4096) {
            w.request_seeds(chunk);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, 1);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
        }
    }
    let wpc = cfg.world_units_per_cell as f64;
    let agents_of = |w: &World| -> (usize, f64, f64) {
        let a: Vec<_> = w.read_agents_blocking(&gpu).into_iter().filter(|a| a.alive != 0).collect();
        let n = a.len().max(1) as f64;
        (a.len(), a.iter().map(|a| a.pos_x as f64).sum::<f64>() / n / wpc, a.iter().map(|a| a.pos_y as f64).sum::<f64>() / n / wpc)
    };
    let gamma_of = |w: &World| w.read_gamma_blocking(&gpu);
    let mono_of = |w: &World| -> Vec<u32> {
        let c = w.read_cells_blocking(&gpu);
        c.chunks(4).map(|q| q.iter().map(|v| (v & 0xFFFF) + (v >> 16)).sum()).collect()
    };
    let (gm0, gx0, gy0) = com(&gamma_of(&w), g, |c| c as u64);
    let (mm0, mx0, my0) = com(&mono_of(&w), g, |c| c as u64);
    println!("início: entulho {gm0} grãos, centro ({gx0:.2}, {gy0:.2}); monómeros {mm0}, centro ({mx0:.2}, {my0:.2})");
    let steps = envf("STEPS").unwrap_or(20000.0) as u32;
    let report_every = (steps / 4).max(64);
    let mut done = 0;
    while done < steps {
        let k = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += k;
        if done % report_every < k || done == steps {
            let (gm, gx, gy) = com(&gamma_of(&w), g, |c| c as u64);
            let (mm, mx, my) = com(&mono_of(&w), g, |c| c as u64);
            if n_agents > 0 {
                let (na, ax, ay) = agents_of(&w);
                println!("        agentes {na}, centro ({ax:.1}, {ay:.1}) (o meio é {})", g / 2);
            }
            println!(
                "{done:6} passos: entulho Δ({:+.3}, {:+.3}) [{gm}]   monómeros Δ({:+.3}, {:+.3}) [{mm}]",
                gx - gx0,
                gy - gy0,
                mx - mx0,
                my - my0
            );
        }
    }
}
