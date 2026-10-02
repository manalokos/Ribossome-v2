//! Observação: estatísticas da população e parentesco genético (8-meros
//! canónicos: o complementar invertido conta como o mesmo genoma).

use ribossome::gpu::Gpu;
use ribossome::params::{SpawnRequest, WorldConfig};
use ribossome::world::World;

fn bases(s: &str) -> Vec<u8> {
    s.chars().map(|c| "AUGC".find(c).unwrap() as u8).collect()
}

#[test]
fn kinship_and_stats() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 4);
    world.seed_matter(&gpu, 4);
    world.params.death_probability = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.maintenance_cost = 0.0;
    let g = bases("AUGGCUAAAGGCUUCGAUCCGGAAUUCGGAUCCUUAGGCAUCGAUGCAUUAA");
    let rc: Vec<u8> = g.iter().rev().map(|b| b ^ 1).collect();
    let other = bases("AUGCCCGGGUUUAAACCCGGGUUUAAACCCGGGUUUAAACCCGGGUUUAA");
    let s = cfg.sim_size();
    world.request_seeds(&[
        SpawnRequest::with_genome(s * 0.3, s * 0.5, &g),
        SpawnRequest::with_genome(s * 0.5, s * 0.5, &rc),
        SpawnRequest::with_genome(s * 0.7, s * 0.5, &other),
    ]);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_steps(&gpu.queue, &mut enc, 2);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();

    world.set_kin_target(&gpu.queue, &g);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_kinship(&mut enc);
    assert!(world.encode_stats(&mut enc));
    gpu.queue.submit([enc.finish()]);
    world.stats_after_submit();
    gpu.wait_idle();

    let agents = world.read_agents_blocking(&gpu);
    let kin = world.read_f32_blocking(&gpu, &world.kin_buf);
    let mut found: Vec<(u32, f32)> =
        agents.iter().enumerate().filter(|(_, a)| a.alive != 0).map(|(i, a)| (a.gene_len, kin[i])).collect();
    found.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("parentesco: {found:?}");
    assert_eq!(found.len(), 3);
    assert!((found[0].1 - 1.0).abs() < 1e-6 && (found[1].1 - 1.0).abs() < 1e-6, "o próprio e o complementar invertido = 1");
    assert!(found[2].1 < 0.2, "um genoma sem relação partilha poucos 8-meros");

    let w = world.poll_stats(&gpu.device).expect("estatísticas lidas");
    assert_eq!(w[0], 3, "3 vivos");
    let genes: u32 = agents.iter().filter(|a| a.alive != 0).map(|a| a.gene_len).sum();
    assert_eq!(w[3], genes, "soma das bases dos genomas");
}
