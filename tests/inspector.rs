//! O inspetor lê o organismo certo: escolhe um agente vivo pelo clique,
//! lê-o de forma assíncrona e a proteína traduzida no CPU a partir do genoma
//! lido é igual ao corpo que a GPU construiu.

use ribossome::gpu::Gpu;
use ribossome::life::organs::{organ_byte, translate_organs};
use ribossome::params::WorldConfig;
use ribossome::ui::inspector::Inspector;
use ribossome::world::World;

#[test]
fn inspector_reads_genome_and_body() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 5);
    world.seed_matter(&gpu, 5);
    let mut rng = ribossome::life::SplitMix(1);
    world.request_seeds(&ribossome::life::seed_requests(200, [30, 120], true, cfg.sim_size(), &mut rng));
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_steps(&gpu.queue, &mut enc, 10);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();

    let target = *world.read_agents_blocking(&gpu).iter().find(|a| a.alive != 0 && a.body_len > 5).expect("agente");
    let mut ins = Inspector::new(&gpu, &world);
    ins.pick(&gpu, &world, [target.pos_x, target.pos_y]);
    for _ in 0..10 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        ins.encode(&world, &mut enc);
        gpu.queue.submit([enc.finish()]);
        ins.after_submit();
        gpu.wait_idle();
        ins.poll(&gpu.device);
        if ins.data.is_some() {
            break;
        }
    }
    let d = ins.data.as_ref().expect("o inspetor não leu nada");
    assert_eq!(d.genome.len() as u32, d.agent.gene_len);
    let cpu = translate_organs(&d.genome, world.params.require_start != 0);
    let aa: Vec<u8> = cpu.iter().map(|r| r.aa).collect();
    let organs: Vec<u8> = cpu.iter().map(organ_byte).collect();
    assert_eq!(aa, d.body, "proteína do CPU ≠ corpo da GPU");
    assert_eq!(organs, d.organs, "órgãos do CPU ≠ órgãos da GPU");
}

/// A tradução com órgãos da GPU é igual à do CPU em TODOS os agentes vivos
/// (e há órgãos na população, para o teste valer alguma coisa).
#[test]
fn gpu_translation_matches_cpu_everywhere() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 6);
    world.seed_matter(&gpu, 6);
    let mut rng = ribossome::life::SplitMix(2);
    world.request_seeds(&ribossome::life::seed_requests(600, [30, 200], true, cfg.sim_size(), &mut rng));
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_steps(&gpu.queue, &mut enc, 5);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let words = |b: &wgpu::Buffer| -> Vec<u32> { bytemuck::cast_slice(&gpu.read_buffer_blocking(b)).to_vec() };
    let (gw, bw, ow) = (words(&world.genomes_buf), words(&world.bodies_buf), words(&world.organs_buf));
    let mut checked = 0;
    let mut organs_seen = 0;
    for (s, a) in world.read_agents_blocking(&gpu).iter().enumerate().filter(|(_, a)| a.alive != 0) {
        let genome: Vec<u8> =
            (0..a.gene_len as usize).map(|i| ((gw[s * 16 + i / 16] >> ((i % 16) * 2)) & 3) as u8).collect();
        let byte = |w: &[u32], i: usize| ((w[s * 16 + i / 4] >> ((i % 4) * 8)) & 0xFF) as u8;
        let cpu = translate_organs(&genome, world.params.require_start != 0);
        assert_eq!(cpu.len() as u32, a.body_len, "comprimento do corpo difere (slot {s})");
        for (k, r) in cpu.iter().enumerate() {
            assert_eq!(r.aa, byte(&bw, k), "aminoácido difere (slot {s}, resíduo {k})");
            assert_eq!(organ_byte(r), byte(&ow, k), "órgão difere (slot {s}, resíduo {k})");
            organs_seen += r.organ.is_some() as usize;
        }
        checked += 1;
    }
    eprintln!("{checked} agentes verificados, {organs_seen} órgãos");
    assert!(checked > 100 && organs_seen > 50);
}
