//! ONDE ESTÁ A ONDA? Um nadador (relógio + TAIL glicinas) parado no
//! laboratório: amplitude do sinal α e do movimento LATERAL de cada resíduo
//! (no referencial do corpo) ao longo da cadeia, durante 3 períodos do
//! relógio. Num flagelo a amplitude cresce para a ponta da cauda; aqui vê-se
//! se é a cabeça ou a cauda que abana. Modos de sinais 0 e 2.
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let tail: usize = std::env::var("TAIL").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    // UTR=1: com bases não traduzidas nas duas pontas (os fios de RNA), para ver a fita mole.
    let utr = std::env::var("UTR").is_ok();
    let (lead, trail) = if utr { ("GCC GCA GCC GCA GCC ", " GCU CGC UCG CUC GCU CGC UCG CUC GCU CGC") } else { ("", "") };
    let genome = bases(&format!("{lead}AUG CAU CUU GAA {} UAA{trail}", "GGU ".repeat(tail)));
    // GENOME="AUG ... UAA ... AUG ... UAA": um genoma à escolha (por exemplo com dois genes).
    let genome = std::env::var("GENOME").map(|g| bases(&g)).unwrap_or(genome);
    {
        let w0 = World::new(&gpu, cfg, 3);
        let code = ribossome::life::table::code_to_gpu(&w0.organ_code);
        let body = ribossome::life::organs::translate_organs(&genome, true, &code);
        let txt: String = body.iter().map(|r| r.organ.map_or(ribossome::life::amino::AA_LETTERS[r.aa as usize], |(t, _, _)| ribossome::life::organs::ORGAN_SYMBOLS[t as usize])).collect();
        println!("tradução no CPU: {} resíduos: {txt}", body.len());
    }
    for mode in [0.0f32, 2.0] {
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
        w.params.thermal_kt = 0.0;
        w.params.spawn_energy = 60.0;
        w.params.signal_mode = mode;
        w.request_seeds(&[SpawnRequest::with_genome(8000.0, 8000.0, &genome)]);
        let step = |w: &mut World, k: u32| {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
        };
        // Deixa assentar (dobragem ao nascer) antes de medir.
        for _ in 0..4 {
            step(&mut w, 64);
        }
        let agents = w.read_agents_blocking(&gpu);
        let Some((slot, a)) = agents.iter().enumerate().find(|(_, a)| a.alive != 0) else {
            println!("modo {mode}: o nadador não nasceu");
            continue;
        };
        let n = a.body_len as usize;
        let (mut smin, mut smax) = (vec![f32::MAX; n], vec![f32::MIN; n]);
        let (mut ymin, mut ymax) = (vec![f32::MAX; n], vec![f32::MIN; n]);
        let start = [a.pos_x, a.pos_y];
        for _ in 0..210 {
            step(&mut w, 1);
            let sig: Vec<[f32; 4]> = bytemuck::cast_slice(&gpu.read_ranges_blocking(&w.signals_buf, &[(slot as u64 * 64 * 16, 64 * 16)])).to_vec();
            let pos: Vec<[f32; 2]> = bytemuck::cast_slice(&gpu.read_ranges_blocking(&w.body_pos_buf, &[(slot as u64 * 64 * 8, 64 * 8)])).to_vec();
            // Eixo do corpo: do primeiro ao último resíduo; lateral = distância a esse eixo.
            let ax = [pos[n - 1][0] - pos[0][0], pos[n - 1][1] - pos[0][1]];
            let len = (ax[0] * ax[0] + ax[1] * ax[1]).sqrt().max(1e-6);
            for k in 0..n {
                smin[k] = smin[k].min(sig[k][0]);
                smax[k] = smax[k].max(sig[k][0]);
                let d = [pos[k][0] - pos[0][0], pos[k][1] - pos[0][1]];
                let lat = (ax[0] * d[1] - ax[1] * d[0]) / len;
                ymin[k] = ymin[k].min(lat);
                ymax[k] = ymax[k].max(lat);
            }
        }
        let end = w.read_agents_blocking(&gpu)[slot];
        let moved = ((end.pos_x - start[0]).powi(2) + (end.pos_y - start[1]).powi(2)).sqrt();
        println!("modo dos sinais {mode}: corpo de {n} resíduos (0 = M, 1 = relógio), avançou {moved:.1} em 210 passos");
        println!("  resíduo:            {}", (0..n).map(|k| format!("{k:5}")).collect::<String>());
        println!("  amplitude do sinal α{}", (0..n).map(|k| format!("{:5.2}", (smax[k] - smin[k]) / 2.0)).collect::<String>());
        println!("  vaivém lateral      {}", (0..n).map(|k| format!("{:5.1}", (ymax[k] - ymin[k]) / 2.0)).collect::<String>());
        if let Ok(out) = std::env::var("OUT") {
            let cap = ribossome::render::capture::Capture::new(&gpu, &w, 384);
            for shot in 0..3 {
                step(&mut w, 9);
                let b = w.read_agents_blocking(&gpu)[slot];
                cap.view.focus.set(slot as u32);
                cap.view.focus_offset.set([0.0, 0.0]);
                let rgba = cap.render(&gpu, &w, &ribossome::render::Camera { center: [b.pos_x, b.pos_y], zoom: 384.0 / 520.0 }, 0, 0.0);
                cap.save_png(&rgba, std::path::Path::new(&format!("{out}_{}_{shot}.png", mode as u32))).unwrap();
            }
        }
    }
}
