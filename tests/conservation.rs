//! Testes do mundo sem UI. Precisam de uma GPU (adaptador por omissão, sem janela).
//!
//! Conservação: mundo pequeno, N passos com TODOS os sistemas do mundo
//! ligados (fluido, temperatura, luz, reações, transporte). A matéria tem de
//! ser constante AO QUANTUM, no total e por canal (as reações mudam o
//! estado ativado/gasto, mas nunca o tipo nem a quantidade). A contagem de
//! referência é feita no CPU a partir da grelha inteira lida da GPU; o
//! livro-razão da GPU é comparado com ela.

use ribossome::gpu::Gpu;
use ribossome::params::WorldConfig;
use ribossome::world::{Ledger, MAX_STEPS_PER_FRAME, World};

fn run(gpu: &Gpu, world: &mut World, steps: u32) {
    let mut done = 0;
    while done < steps {
        let n = MAX_STEPS_PER_FRAME.min(steps - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        world.encode_steps(&gpu.queue, &mut enc, n);
        gpu.queue.submit([enc.finish()]);
        gpu.wait_idle();
        done += n;
    }
}

fn channels(l: &Ledger) -> [u64; 4] {
    [l.channel(0), l.channel(1), l.channel(2), l.channel(3)]
}

#[test]
fn matter_is_conserved_exactly() {
    const STEPS: u32 = 4096;
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 7);
    // Difusão forte para haver muitos saltos (e corridas entre threads).
    world.params.diffusion = 30.0;
    world.params.settle = 1.0;
    let seeded = world.seed_matter(&gpu, 7);
    assert!(seeded.total() > 0);

    let initial_cells = world.read_cells_blocking(&gpu);
    assert_eq!(Ledger::from_cells(&initial_cells), seeded, "a sementeira não chegou intacta à GPU");
    assert_eq!(world.ledger_blocking(&gpu), seeded, "livro-razão da GPU ≠ contagem do CPU no arranque");

    run(&gpu, &mut world, STEPS);

    let cells = world.read_cells_blocking(&gpu);
    let after = Ledger::from_cells(&cells);
    let moved = cells.iter().zip(&initial_cells).filter(|(a, b)| a != b).count();
    let act_before: u64 = seeded.act.iter().map(|&v| v as u64).sum();
    let act_after: u64 = after.act.iter().map(|&v| v as u64).sum();
    eprintln!(
        "{STEPS} passos em {}²: total {} → {}, ativados {act_before} → {act_after}, {moved} de {} células×canal mudaram",
        cfg.grid_size,
        seeded.total(),
        after.total(),
        cells.len()
    );
    assert!(moved > cells.len() / 10, "quase nada se moveu: o teste não estaria a testar nada");
    assert_ne!(act_before, act_after, "nenhuma reação mudou o estado: as reações não estão a correr");
    assert_eq!(after.total(), seeded.total(), "matéria total não conservada");
    assert_eq!(channels(&after), channels(&seeded), "matéria não conservada por canal");
    assert_eq!(world.ledger_blocking(&gpu), after, "livro-razão da GPU ≠ contagem do CPU");
}

/// O fluido arranca da fumarola, fica finito, e a luz UV cai com a profundidade.
#[test]
fn fluid_and_light_are_sane() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 3);
    world.seed_matter(&gpu, 3);
    run(&gpu, &mut world, 600);

    let vel = world.read_f32_blocking(&gpu, &world.velocity_buf);
    let temp = world.read_f32_blocking(&gpu, &world.temp_buf);
    assert!(vel.iter().chain(&temp).all(|v| v.is_finite()), "NaN/Inf no fluido");
    let max_speed = vel.chunks(2).map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt()).fold(0.0f32, f32::max);
    let max_t = temp.iter().cloned().fold(0.0f32, f32::max);
    // Velocidade média vertical na metade de cima da pluma (deve subir: +y).
    let n = cfg.fluid_size as usize;
    let mean_vy: f32 = vel.chunks(2).map(|v| v[1]).sum::<f32>() / (n * n) as f32;
    eprintln!("fluido: |v|max {max_speed:.3} células/s, T max {max_t:.2}, vy média {mean_vy:.4}");
    assert!(max_t > 2.0, "a fumarola não aqueceu a água");
    assert!(max_speed > 0.1, "a flutuação não pôs o fluido em movimento");

    // Incompressibilidade num aquário fechado: o fluxo líquido através de
    // qualquer linha horizontal tem de ser ~0 (com Jacobi 10 era ~50%).
    for y in [n / 4, n / 2, 3 * n / 4] {
        let net: f32 = (0..n).map(|x| vel[(y * n + x) * 2 + 1]).sum::<f32>();
        let gross: f32 = (0..n).map(|x| vel[(y * n + x) * 2 + 1].abs()).sum::<f32>();
        eprintln!("linha {y}: fluxo líquido {net:.2}, total {gross:.2}");
        assert!(net.abs() < 0.1 * gross.max(n as f32), "o fluido cria/destrói água na linha {y}");
    }

    let light = world.read_f32_blocking(&gpu, &world.light_buf);
    let g = cfg.grid_size as usize;
    let top = light[(g - 1) * g + g / 2];
    let bottom = light[g / 2];
    eprintln!("luz: topo {top:.3}, fundo {bottom:.5}");
    assert!(top > 0.9 && bottom < top * 0.01, "a luz não cai com a profundidade como esperado");
}

/// O resultado é reprodutível bit a bit e não depende de como os passos são
/// agrupados por frame (64 por submissão vs um a um). Física do terreno
/// desligada: o terreno ainda mexe nas células no lugar.
#[test]
fn simulation_is_deterministic() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let go = |batch: u32| {
        let mut world = World::new(&gpu, WorldConfig::TEST, 9);
        world.seed_matter(&gpu, 9);
        world.settings.terrain_enabled = false;
        let mut done = 0;
        while done < 256 {
            let k = batch.min(256 - done);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            world.encode_steps(&gpu.queue, &mut enc, k);
            gpu.queue.submit([enc.finish()]);
            done += k;
        }
        gpu.wait_idle();
        world.read_cells_blocking(&gpu)
    };
    let a = go(64);
    assert_eq!(a, go(64), "duas corridas iguais deram resultados diferentes");
    assert_eq!(a, go(1), "o resultado depende do agrupamento dos passos por frame");
}

/// Ciclo de vida da fase 3a: sementes montadas da sopa, deriva e morte.
/// A matéria total (livre + presa nos agentes) mantém-se ao quantum, por
/// canal, enquanto nascem e morrem agentes.
#[test]
fn life_cycle_conserves_matter() {
    use ribossome::params::SpawnRequest;
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 11);
    let seeded = world.seed_matter(&gpu, 11);
    let s = cfg.sim_size();
    let reqs: Vec<SpawnRequest> = (0..300)
        .map(|i| {
            let f = i as f32 / 300.0;
            SpawnRequest {
                pos_x: s * (0.05 + 0.9 * ((f * 37.0).fract())),
                pos_y: s * (0.3 + 0.65 * ((f * 13.0).fract())),
                gene_len: 6 + (i % 120),
                flags: i % 2,
            }
        })
        .collect();
    world.request_seeds(&reqs);
    run(&gpu, &mut world, 1);
    let after_spawn = world.ledger_blocking(&gpu);
    let lc = world.life_counters_blocking(&gpu);
    eprintln!("sementes: {} nasceram, {} falharam; presos {}", lc.spawned, lc.spawn_failed, after_spawn.held_total());
    assert!(lc.spawned > 100, "quase nenhuma semente nasceu");
    assert_eq!(lc.spawned + lc.spawn_failed, reqs.len() as u32);
    assert_eq!(after_spawn.total(), seeded.total(), "as sementes criaram ou destruíram matéria");

    run(&gpu, &mut world, 2000);
    let end = world.ledger_blocking(&gpu);
    let lc = world.life_counters_blocking(&gpu);
    let alive = world.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).count() as u32;
    eprintln!(
        "2000 passos: {} nascimentos, {} mortes, {alive} vivos, presos {}",
        lc.births,
        lc.deaths,
        end.held_total()
    );
    assert!(lc.deaths > 50, "quase ninguém morreu: a morte não está a ser testada");
    assert_eq!(alive, lc.spawned + lc.births - lc.deaths, "vivos ≠ sementes + nascimentos − mortes");
    assert_eq!(lc.free_top, cfg.max_agents - alive, "a pilha de slots livres não bate certo");
    assert_eq!(end.total(), seeded.total(), "matéria total não conservada com vida");
    assert_eq!(channels(&end), channels(&seeded), "matéria não conservada por canal com vida");
}
