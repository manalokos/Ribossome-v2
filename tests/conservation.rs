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
    // Reações ligadas (os defaults podem tê-las desligadas): ativação térmica
    // e agregação também mexem nos ativados.
    world.params.thermal_activation = 1.0;
    world.params.aggregation = 0.02;
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
    // A queda da luz testa-se com a absorção ligada (os defaults podem tê-la
    // desligada) e sempre de dia.
    world.params.uv_depth = 2.0;
    world.params.monomer_uv_absorb = 1.1;
    world.params.day_period = 0.0;
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
    // A luz está a 1/LIGHT_DIV da resolução da grelha.
    let g = (cfg.grid_size / ribossome::shaders::LIGHT_DIV) as usize;
    let top = light[(g - 1) * g + g / 2];
    let bottom = light[g / 2];
    eprintln!("luz: topo {top:.3}, fundo {bottom:.5}");
    // Os monómeros da primeira linha já absorvem (no mundo de teste cada linha
    // da luz é 1/128 da altura).
    assert!(top > 0.7 && bottom < top * 0.01, "a luz não cai com a profundidade como esperado");
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
            SpawnRequest::new(
                s * (0.05 + 0.9 * ((f * 37.0).fract())),
                s * (0.3 + 0.65 * ((f * 13.0).fract())),
                6 + (i % 120),
                i % 2,
            )
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

/// "Ativar já": muda gastos para ativados sem criar nem destruir matéria.
#[test]
fn activate_spent_conserves_matter() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let mut world = World::new(&gpu, WorldConfig::TEST, 13);
    let before = world.seed_matter(&gpu, 13);
    let spent_before: u64 = before.spent.iter().map(|&v| v as u64).sum();
    let n = world.activate_spent(&gpu, 0.5, 1);
    let after = Ledger::from_cells(&world.read_cells_blocking(&gpu));
    assert_eq!(after.total(), before.total(), "ativar não pode mudar a matéria");
    assert_eq!(channels(&after), channels(&before), "nem por canal");
    let act_gain: u64 = after.act.iter().map(|&v| v as u64).sum::<u64>() - before.act.iter().map(|&v| v as u64).sum::<u64>();
    assert_eq!(act_gain, n, "os ativados ganhos são os que a função diz");
    assert!((n as f64 - spent_before as f64 * 0.5).abs() < spent_before as f64 * 0.02, "~metade dos gastos: {n} de {spent_before}");
}

/// Trocar o terreno do mundo vivo e pintar rocha não criam nem destroem
/// monómeros: os que deixam de caber saem para o lado.
#[test]
fn live_terrain_and_painting_conserve_matter() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let cfg = WorldConfig::TEST;
    let mut world = World::new(&gpu, cfg, 21);
    let before = world.seed_matter(&gpu, 21);
    let n = cfg.grid_size as usize;

    // Terreno novo: um bloco de rocha a meio e uma faixa de entulho.
    let mut gamma = vec![0u32; n * n];
    for y in 0..n {
        for x in 0..n {
            if (n / 4..n / 2).contains(&x) && (n / 4..n / 2).contains(&y) {
                gamma[y * n + x] = 6;
            } else if y < n / 8 {
                gamma[y * n + x] = 2;
            }
        }
    }
    world.apply_terrain_live(&gpu, gamma.clone(), vec![0.0; n * n], None);
    let cells = world.read_cells_blocking(&gpu);
    let after = Ledger::from_cells(&cells);
    assert_eq!(channels(&after), channels(&before), "trocar o terreno mudou a matéria por canal");
    assert_eq!(after.total(), before.total());
    // Nada ficou dentro da rocha nova.
    let in_rock: u64 = (0..n * n)
        .filter(|&i| gamma[i] >= 3)
        .map(|i| (0..4).map(|c| ((cells[i * 4 + c] & 0xFFFF) + (cells[i * 4 + c] >> 16)) as u64).sum::<u64>())
        .sum();
    assert_eq!(in_rock, 0, "ficaram monómeros dentro da rocha carregada");

    // Pintar um disco de rocha noutro sítio e correr uns passos.
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_paint(&gpu.queue, &mut enc, n as f32 * 0.75, n as f32 * 0.75, 10.0, 6);
    gpu.queue.submit([enc.finish()]);
    let g2 = world.read_gamma_blocking(&gpu);
    assert_eq!(g2[(n * 3 / 4) * n + n * 3 / 4], 6, "o pincel não pôs rocha");
    run(&gpu, &mut world, 64);
    let painted = Ledger::from_cells(&world.read_cells_blocking(&gpu));
    assert_eq!(channels(&painted), channels(&before), "pintar rocha mudou a matéria por canal");
}

/// "Recomeçar": o terreno fica exatamente como estava na GPU, não há
/// agentes, a epoch volta a zero e a matéria semeada é a que lá fica.
#[test]
fn restart_keeps_the_terrain() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let mut world = World::new(&gpu, WorldConfig::TEST, 21);
    world.seed_matter(&gpu, 21);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    world.encode_steps(&gpu.queue, &mut enc, 20);
    gpu.queue.submit([enc.finish()]);
    gpu.wait_idle();
    let terrain_before = gpu.read_buffer_blocking(&world.gamma_buf);
    let seeded = world.restart_keeping_terrain(&gpu, 22);
    assert_eq!(world.params.epoch, 0);
    assert_eq!(gpu.read_buffer_blocking(&world.gamma_buf), terrain_before);
    assert_eq!(world.read_agents_blocking(&gpu).iter().filter(|a| a.alive != 0).count(), 0);
    let now = world.ledger_blocking(&gpu);
    assert_eq!(now.total(), seeded.total());
    assert_eq!(now.held_total(), 0);
}

/// O transporte de N em N passos também conserva a matéria e continua a
/// mexer nela.
#[test]
fn sparse_transport_conserves_matter() {
    let gpu = Gpu::new_headless().expect("este teste precisa de uma GPU");
    let mut world = World::new(&gpu, WorldConfig::TEST, 5);
    world.params.diffusion = 10.0;
    world.params.settle = 1.0;
    world.params.thermal_activation = 1.0;
    world.params.aggregation = 0.02;
    world.params.transport_every = 3;
    let seeded = world.seed_matter(&gpu, 5);
    let before = world.read_cells_blocking(&gpu);
    run(&gpu, &mut world, 600);
    let cells = world.read_cells_blocking(&gpu);
    let after = Ledger::from_cells(&cells);
    let moved = cells.iter().zip(&before).filter(|(a, b)| a != b).count();
    assert!(moved > cells.len() / 10, "quase nada se moveu");
    assert_eq!(after.total(), seeded.total(), "matéria total não conservada");
    assert_eq!(channels(&after), channels(&seeded), "matéria não conservada por canal");
}
