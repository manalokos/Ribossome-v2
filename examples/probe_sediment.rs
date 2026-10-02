//! Sedimentação: sem fluido e sem natação, os agentes afundam
//! sedimentation·√n por passo (até ao terreno). Compara a descida medida com
//! a esperada.
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::World;

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::DEFAULT;
    let mut world = World::new(&gpu, cfg, 1);
    world.seed_matter(&gpu, 1);
    world.settings.fluid_enabled = false;
    world.params.rft_enabled = 0;
    world.params.death_probability = 0.0;
    world.params.pairing_rate = 0.0;
    world.params.maintenance_cost = 0.0;
    let mut rng = ribossome::life::SplitMix(3);
    world.request_seeds(&ribossome::life::seed_requests(300, [30, 200], true, cfg.sim_size(), &mut rng));
    let run = |world: &mut World, n: u32| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, n);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
    };
    run(&mut world, 1);
    let a0 = world.read_agents_blocking(&gpu);
    for _ in 0..16 {
        run(&mut world, 64);
    }
    let a1 = world.read_agents_blocking(&gpu);
    let (mut measured, mut expected, mut cnt) = (0.0f64, 0.0f64, 0);
    for (a, b) in a0.iter().zip(&a1) {
        // Só os que estavam bem acima do fundo (não chegaram ao terreno).
        if a.alive != 0 && b.alive != 0 && a.id == b.id && a.pos_y > cfg.sim_size() * 0.5 {
            measured += (a.pos_y - b.pos_y) as f64;
            expected += 0.02 * (a.body_len.max(1) as f64).sqrt() * 1024.0;
            cnt += 1;
        }
    }
    println!("{cnt} agentes: descida média {:.1} (esperada {:.1}) em 1024 passos", measured / cnt as f64, expected / cnt as f64);
}
