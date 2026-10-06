//! O dano UV e o protetor solar. Corpos DESENHADOS junto à superfície, em
//! sol pleno e sempre de dia, sem comer nem pagar manutenção: só morrem pela
//! mortalidade base e pelo UV. Compara 12 valinas (sem proteção) com corpos
//! em que 1, 2 ou 4 das 12 são triptofano, e com o dano UV desligado.
use std::collections::HashMap;

use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn bases(s: &str) -> Vec<u8> {
    s.chars().map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

fn main() {
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
    let designs: Vec<(String, String)> = [0usize, 1, 2, 4]
        .iter()
        .map(|&w| (format!("{w} triptofanos em 12"), format!("AUG{}{}UAA", "UGG".repeat(w), "GUU".repeat(12 - w))))
        .collect();
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let steps = envf("STEPS", 300.0) as u32;
    for damage in [envf("DAMAGE", 30.0), 1.0] {
        let mut w = World::new(&gpu, cfg, 3);
        let n = cfg.cells() as usize;
        w.custom_terrain = Some((vec![0; n], vec![0.0; n]));
        w.fumaroles.clear();
        w.seed_matter(&gpu, 3);
        w.settings.fluid_enabled = false;
        w.settings.contact_enabled = false;
        w.params.day_period = 0.0;
        w.params.uv_strength = 3.0;
        w.params.uv_depth = 0.0;
        w.params.monomer_uv_absorb = 0.0;
        w.params.uv_damage = damage;
        w.params.maintenance_cost = 0.0;
        w.params.motion_cost = 0.0;
        w.params.pairing_rate = 0.0;
        w.params.uptake_rate = 0.0;
        w.params.sedimentation = 0.0;
        w.params.death_metab = 0.0;
        w.params.spawn_energy = 60.0;
        let mut rng = ribossome::life::SplitMix(5);
        let s = cfg.sim_size();
        let mut reqs = Vec::new();
        let mut by_genome: HashMap<Vec<u8>, usize> = HashMap::new();
        for (i, (_, g)) in designs.iter().enumerate() {
            let b = bases(g);
            by_genome.insert(b.clone(), i);
            for _ in 0..300 {
                reqs.push(SpawnRequest::with_genome(s * (0.05 + 0.9 * rng.f32()), s * (0.9 + 0.08 * rng.f32()), &b));
            }
        }
        w.request_seeds(&reqs);
        let count = |w: &World| -> Vec<u32> {
            let words: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&w.genomes_buf)).to_vec();
            let mut out = vec![0u32; designs.len()];
            for (slot, a) in w.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
                let g: Vec<u8> = (0..a.gene_len as usize).map(|i| ((words[slot * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
                if let Some(&i) = by_genome.get(&g) {
                    out[i] += 1;
                }
            }
            out
        };
        let run = |w: &mut World, k: u32| {
            let mut done = 0;
            while done < k {
                let c = MAX_STEPS_PER_FRAME.min(k - done);
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                w.encode_steps(&gpu.queue, &mut enc, c);
                gpu.queue.submit([enc.finish()]);
                gpu.wait_idle();
                done += c;
            }
        };
        run(&mut w, 20);
        let c0 = count(&w);
        run(&mut w, steps);
        let c1 = count(&w);
        print!("dano UV {damage}: sobrevivem em {steps} passos:");
        for (i, (name, _)) in designs.iter().enumerate() {
            print!("  {name}: {:.0}% ({} de {})", 100.0 * c1[i] as f32 / c0[i].max(1) as f32, c1[i], c0[i]);
        }
        println!();
    }
}
