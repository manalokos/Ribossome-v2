//! Dinâmica da população com as regras normais (mundo grande): vivos,
//! nascimentos, mortes, energia, idade, geração, órgãos e comida livre.
//! Variáveis: STEPS (total), EVERY (intervalo), SEEDS, CAP (capacidade).
use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let lab = env("LAB", 0) == 1;
    let mut cfg = WorldConfig { max_agents: env("CAP", 60_000), ..WorldConfig::DEFAULT };
    if lab {
        cfg.grid_size = 1024;
        cfg.fluid_size = 512;
    }
    let mut world = World::new(&gpu, cfg, 1);
    if lab {
        world.configure_lab();
        world.seed_lab(&gpu, 1, env("FOOD", 6.0));
        world.params.reactivation_rate = env("REACT", world.params.reactivation_rate);
    } else {
        world.seed_matter(&gpu, 1);
    }
    world.params.maintenance_cost = env("MAINT", world.params.maintenance_cost);
    world.params.uptake_rate = env("UPTAKE", world.params.uptake_rate);
    world.params.pairing_cost = env("PCOST", world.params.pairing_cost);
    let mut rng = ribossome::life::SplitMix(11);
    world.request_seeds(&ribossome::life::seed_requests(env("SEEDS", 3000), [30, 200], true, cfg.sim_size(), &mut rng));
    let total: u32 = env("STEPS", 10_000);
    let every: u32 = env("EVERY", 1000);
    let (mut last_b, mut last_d, mut last_s) = (0u32, 0u32, 0u32);
    println!("passo   vivos  nasc.  mortes  fome%  E/cap  idade  ger.máx  c/relógio  c/sensor  ativ.livres  presos%");
    let mut done = 0;
    while done < total {
        let mut k = 0;
        while k < every {
            let n = MAX_STEPS_PER_FRAME.min(every - k);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            world.encode_steps(&gpu.queue, &mut enc, n);
            gpu.queue.submit([enc.finish()]);
            gpu.wait_idle();
            k += n;
        }
        done += every;
        let agents = world.read_agents_blocking(&gpu);
        let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&world.organs_buf)).to_vec();
        let alive: Vec<(usize, _)> = agents.iter().enumerate().filter(|(_, a)| a.alive != 0).collect();
        let n = alive.len().max(1) as f32;
        let e_frac = alive.iter().map(|(_, a)| a.energy / (a.body_len.max(1) as f32)).sum::<f32>() / n;
        let age = alive.iter().map(|(_, a)| a.age as f32).sum::<f32>() / n;
        let max_gen = alive.iter().map(|(_, a)| a.generation).max().unwrap_or(0);
        let clocks = alive
            .iter()
            .filter(|(s, a)| {
                (0..a.body_len as usize).any(|k| {
                    let o = (organs[s * 32 + k / 2] >> ((k % 2) * 16)) & 0xFF;
                    o != 0 && (o & 0xF) - 1 == 5
                })
            })
            .count();
        let sensors = alive
            .iter()
            .filter(|(s, a)| {
                (0..a.body_len as usize).any(|k| {
                    let o = (organs[s * 32 + k / 2] >> ((k % 2) * 16)) & 0xFF;
                    o != 0 && matches!((o & 0xF) - 1, 2..=4)
                })
            })
            .count();
        let lc = world.life_counters_blocking(&gpu);
        let l = world.ledger_blocking(&gpu);
        let act: u64 = l.act.iter().map(|&v| v as u64).sum();
        println!(
            "{done:>6} {:>7} {:>6} {:>7} {:>5.0}% {e_frac:>6.2} {age:>6.0} {max_gen:>8} {:>9.1}% {:>8.1}% {act:>11} {:>7.1}",
            alive.len(),
            lc.births - last_b,
            lc.deaths - last_d,
            (lc.starved - last_s) as f32 / (lc.deaths - last_d).max(1) as f32 * 100.0,
            clocks as f32 / n * 100.0,
            sensors as f32 / n * 100.0,
            l.held_total() as f64 / l.total() as f64 * 100.0
        );
        last_b = lc.births;
        last_d = lc.deaths;
        last_s = lc.starved;
    }
}
