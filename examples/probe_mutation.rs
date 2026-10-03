//! Mutações com taxa alta (MUT, por omissão 0.05): a matéria tem de se
//! conservar ao quantum com indels de 1–3 bases e duplicações, e os genomas
//! têm de variar de tamanho (incluindo comprimentos fora dos múltiplos de 3).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig::TEST;
    let mut w = World::new(&gpu, cfg, 5);
    let seeded = w.seed_matter(&gpu, 5);
    w.params.mutation_rate = std::env::var("MUT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
    w.params.reactivation_rate = 0.002;
    w.params.death_probability = 0.0;
    w.params.maintenance_cost = 0.0;
    w.params.pairing_cost = 0.0;
    w.params.spawn_energy = 50.0;
    w.params.heat_kill = 0.0;
    let mut rng = ribossome::life::SplitMix(3);
    w.request_seeds(&ribossome::life::seed_requests(400, [30, 90], true, cfg.sim_size(), &mut rng));
    let mut done = 0;
    while done < 6000 {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        w.encode_steps(&gpu.queue, &mut enc, MAX_STEPS_PER_FRAME);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += MAX_STEPS_PER_FRAME;
    }
    let l = w.ledger_blocking(&gpu);
    let agents = w.read_agents_blocking(&gpu);
    let lens: Vec<u32> = agents.iter().filter(|a| a.alive != 0).map(|a| a.gene_len).collect();
    let c = w.life_counters_blocking(&gpu);
    let mod3: Vec<usize> = (0..3).map(|r| lens.iter().filter(|&&x| x % 3 == r).count()).collect();
    println!("matéria {} -> {} ({})", seeded.total(), l.total(), if seeded.total() == l.total() { "exata" } else { "ERRO" });
    println!(
        "{} vivos, {} nascimentos; genoma min {} máx {}; resto por 3: {:?}",
        lens.len(),
        c.births,
        lens.iter().min().unwrap_or(&0),
        lens.iter().max().unwrap_or(&0),
        mod3
    );
}
