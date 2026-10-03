//! Balanço de energia dos fotossintéticos: agentes desenhados (3 fotossistemas
//! produtores, sem boca) a várias alturas, mundo completo SEM terreno, sem
//! reprodução nem morte ao acaso. Energia por passo = (E fim − E início)/passos.
//! GAIN = codão da intensidade (CCC = ×14,7, GGG = ×2,4, sem = ×1).
use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let gain = std::env::var("GAIN").unwrap_or_default();
    let cfg = WorldConfig::DEFAULT;
    let mut w = World::new(&gpu, cfg, 2);
    let n = cfg.cells() as usize;
    w.custom_terrain = Some((vec![0; n], vec![0.0; n]));
    w.seed_matter(&gpu, 2);
    w.params.death_probability = 0.0;
    w.params.pairing_rate = 0.0;
    w.params.spawn_energy = 10.0;
    let envf = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    if let Some(v) = envf("YIELD") {
        w.params.photo_yield = v;
    }
    if let Some(v) = envf("ABS") {
        w.params.monomer_uv_absorb = v;
    }
    // C (UGU) + S (UCU) = fotossistema A (eficiência 1, produtor).
    let text = format!("AUG {} {} UAA", format!("UGU UCU {gain} GGU ").repeat(3), "GGU ".repeat(8));
    let g: Vec<u8> = text.chars().filter(|c| !c.is_whitespace()).map(|c| "AUGC".find(c).unwrap() as u8).collect();
    let s = cfg.sim_size();
    let heights = [0.97f32, 0.9, 0.75, 0.5, 0.25];
    let mut reqs = Vec::new();
    for (hi, &h) in heights.iter().enumerate() {
        for i in 0..200 {
            let x = s * (0.05 + 0.9 * (i as f32 + 0.5) / 200.0);
            let _ = hi;
            reqs.push(SpawnRequest::with_genome(x, s * h, &g));
        }
    }
    w.request_seeds(&reqs);
    let run = |w: &mut World, steps: u32| {
        let mut done = 0;
        while done < steps {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            w.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            done += MAX_STEPS_PER_FRAME;
        }
    };
    run(&mut w, 64);
    let a0 = w.read_agents_blocking(&gpu);
    let steps = 1000;
    run(&mut w, steps);
    let a1 = w.read_agents_blocking(&gpu);
    let e0: std::collections::HashMap<u32, (f32, f32)> =
        a0.iter().filter(|a| a.alive != 0).map(|a| (a.id, (a.energy, a.pos_y))).collect();
    let body = a0.iter().find(|a| a.alive != 0).map(|a| a.body_len).unwrap_or(0);
    println!(
        "corpo {body} resíduos, intensidade '{gain}', sol {}, profundidade ótica {}, absorção monómeros {}, rendimento {}",
        w.params.uv_strength, w.params.uv_depth, w.params.monomer_uv_absorb, w.params.photo_yield
    );
    let light = w.read_f32_blocking(&gpu, &w.light_buf);
    let ls = (cfg.grid_size / ribossome::shaders::LIGHT_DIV) as usize;
    let mono = w.read_cells_blocking(&gpu);
    for &h in &heights {
        let row = ((h * ls as f32) as usize).min(ls - 1);
        let lmean = light[row * ls..(row + 1) * ls].iter().sum::<f32>() / ls as f32;
        let gy = ((h * cfg.grid_size as f32) as usize).min(cfg.grid_size as usize - 1);
        let g = cfg.grid_size as usize;
        let m: u32 = (0..g).map(|x| (0..4).map(|c| { let v = mono[(gy * g + x) * 4 + c]; (v & 0xFFFF) + (v >> 16) }).sum::<u32>()).sum();
        println!("  altura {h:.2}: luz {lmean:.3}, monómeros por célula {:.1}", m as f32 / g as f32);
    }
    for &h in &heights {
        let d: Vec<f32> = a1
            .iter()
            .filter(|a| a.alive != 0)
            .filter_map(|a| e0.get(&a.id).filter(|(_, y)| (y / s - h).abs() < 0.02).map(|_| a.energy))
            .collect();
        let alive = d.len();
        let mean = d.iter().sum::<f32>() / alive.max(1) as f32;
        println!("  altura {h:.2}: {alive:3}/200 vivos ao fim de {steps} passos, energia média {mean:.1}");
    }
}
