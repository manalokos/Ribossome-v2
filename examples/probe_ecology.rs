//! Ecologia: mundo completo com o terreno do projeto, sementes ao acaso.
//! De EVERY em EVERY passos: população, comida ativada, nascimentos,
//! mortes, mordidas e fração de agentes com cada órgão (fotossistema
//! produtor/reciclador, protease, boca). STEPS, EVERY, SEEDS, REACT.
use ribossome::gpu::Gpu;
use ribossome::life::organs::ORGAN_TYPES;
use ribossome::params::WorldConfig;
use ribossome::world::{MAX_STEPS_PER_FRAME, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let gpu = Gpu::new_headless().unwrap();
    let cfg = WorldConfig { max_agents: env("CAP", 200_000), ..WorldConfig::DEFAULT };
    let mut world = World::new(&gpu, cfg, 1);
    let t: String = env("TERRAIN", String::from("assets/terreno.png"));
    if t != "-" {
        world.load_terrain_png(std::path::Path::new(&t)).unwrap();
    }
    world.seed_matter(&gpu, 1);
    world.params.reactivation_rate = env("REACT", 2e-5f32);
    let mut rng = ribossome::life::SplitMix(7);
    world.request_seeds(&ribossome::life::seed_requests(env("SEEDS", 3000), [30, 200], true, cfg.sim_size(), &mut rng));
    let total: u32 = env("STEPS", 60_000);
    let every: u32 = env("EVERY", 6000);
    let mut done = 0;
    println!("passo   vivos  ativ%  nasc  mortes  mordidas | fotoE  fotoR  protease  boca  (frações dos vivos)");
    while done < total {
        let k = MAX_STEPS_PER_FRAME.min(total - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, k);
        gpu.queue.submit([enc.finish()]);
        done += k;
        if done % every == 0 {
            gpu.wait_idle();
            let c = world.life_counters_blocking(&gpu);
            let l = world.ledger_blocking(&gpu);
            let agents = world.read_agents_blocking(&gpu);
            let organs: Vec<u32> = bytemuck::cast_slice(&gpu.read_buffer_blocking(&world.organs_buf)).to_vec();
            let (mut alive, mut photo_e, mut photo_r, mut prot, mut mouth) = (0u32, 0u32, 0u32, 0u32, 0u32);
            for (s, a) in agents.iter().enumerate() {
                if a.alive == 0 { continue; }
                alive += 1;
                let (mut pe, mut pr, mut pt, mut mo) = (false, false, false, false);
                for kk in 0..a.body_len as usize {
                    let o = (organs[s * 32 + kk / 2] >> ((kk % 2) * 16)) & 0xFFFF;
                    if o == 0 { continue; }
                    let ty = (o & 0xF) as usize - 1;
                    let p = (o >> 4) & 0xF;
                    assert!(ty < ORGAN_TYPES);
                    match ty {
                        10 => {
                            // Reciclador se a variante usar mais de metade da luz para reativar.
                            let rec = world.organ_table[10].variantes[(p as usize).min(5)].get("reciclar").copied().unwrap_or(0.0);
                            if rec > 0.5 { pr = true } else { pe = true }
                        }
                        11 => pt = true,
                        0 => mo = true,
                        _ => {}
                    }
                }
                photo_e += pe as u32; photo_r += pr as u32; prot += pt as u32; mouth += mo as u32;
            }
            let act: u64 = l.act.iter().map(|&v| v as u64).sum();
            let f = |x: u32| x as f64 / alive.max(1) as f64 * 100.0;
            println!(
                "{done:>6} {alive:>7} {:5.1} {:>6} {:>7} {:>9} | {:5.1}% {:5.1}% {:7.1}% {:5.1}%",
                act as f64 / l.free_total() as f64 * 100.0, c.births, c.deaths, c.bites, f(photo_e), f(photo_r), f(prot), f(mouth)
            );
        }
    }
}
