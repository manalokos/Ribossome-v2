//! O inspetor lê o organismo certo: escolhe um agente vivo pelo clique,
//! lê-o de forma assíncrona e a proteína traduzida no CPU a partir do genoma
//! lido é igual ao corpo que a GPU construiu.

use ribossome::gpu::Gpu;
use ribossome::life::amino::translate;
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
    assert_eq!(translate(&d.genome), d.body, "proteína do CPU ≠ corpo da GPU");
}
